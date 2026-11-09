"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    # 第3回 宿題: 承認が必要な save_learning_log まで呼ぶと実行が承認待ちで止まるため、
    # 「全部呼ぶ」から承認なしで動くツールだけを呼ぶ形に変更（承認の流れは下の宿題テストで確認）
    no_approval = [t.name for t in agent._function_toolset.tools.values() if not t.requires_approval]
    model = TestModel(call_tools=no_approval)
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


# --- 第3回 宿題（実装）: save_learning_log と承認 ---

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.tools import ToolDenied

from app import tools
from app.agent import run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """本物の learning_log/ を汚さないよう、保存先を一時フォルダに差し替える。"""
    d = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", d)
    return d


def test_save_learning_log_stops_for_approval(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("今日学んだことを記録して")
    # 承認前なので実行は止まり、承認待ちの呼び出しが返る。ファイルはまだ書かれない
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not log_dir.exists()


def test_save_learning_log_writes_after_approval(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    seen = []

    def approve_all(requests):
        seen.extend(c.tool_name for c in requests.approvals)
        from pydantic_ai import DeferredToolResults
        return DeferredToolResults(approvals={c.tool_call_id: True for c in requests.approvals})

    result = run_with_approval(agent, "今日学んだことを記録して", model, approve=approve_all)
    assert seen == ["save_learning_log"]
    assert isinstance(result.output, str)
    files = list(log_dir.glob("*.md"))
    assert len(files) == 1 and files[0].read_text(encoding="utf-8").startswith("## ")


def test_save_learning_log_is_not_written_when_denied(log_dir):
    model = TestModel(call_tools=["save_learning_log"])

    def deny_all(requests):
        from pydantic_ai import DeferredToolResults
        return DeferredToolResults(
            approvals={c.tool_call_id: ToolDenied("利用者が保存を断りました。") for c in requests.approvals}
        )

    result = run_with_approval(agent, "今日学んだことを記録して", model, approve=deny_all)
    assert isinstance(result.output, str)
    assert not log_dir.exists()
    # 断られたことが、ツールの結果としてモデルに伝わっている
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert any("断りました" in str(p.content) for p in returns)
