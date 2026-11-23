"""第5回 宿題（発展）: Tree のコンパクションのテスト。

FunctionModel（台本どおりにツールを呼ぶ偽モデル）で、複数ページの質問を想定した
同じ順路を --tree と --tree-compact の両方でたどらせ、モデルに届いた入力を比べる。
APIキーは使わない。
"""

import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app import agent as agent_module
from app.compaction import COMPACTED_KEY
from app.tools import browse_tree

# 「コンパクションとセッションをまたぐメモリの違いは？」の根拠（第3回 p.13, 39, 40）へ向かう順路
ROUTE = [None, "lec03-main", "lec03-main-s005", "lec03-main-p013",
         "lec03-main-s037", "lec03-main-p039", "lec03-main-p040"]
PAGE_IDS = {"lec03-main-p013", "lec03-main-p039", "lec03-main-p040"}


def _scripted_model(seen: list[str]) -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        seen.append("".join(
            part.content for m in messages for part in m.parts
            if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
        ))
        step = len(seen) - 1
        if step < len(ROUTE):
            args = {} if ROUTE[step] is None else {"node_id": ROUTE[step]}
            return ModelResponse(parts=[ToolCallPart("browse_tree", args)])
        return ModelResponse(parts=[TextPart("回答（第3回・PDF p.13, 39, 40）")])
    return FunctionModel(respond)


def _run(target):
    seen: list[str] = []
    with target.override(instructions=agent_module.TREE_INSTRUCTIONS, tools=[browse_tree]):
        result = target.run_sync("質問", model=_scripted_model(seen))
    return result, seen


def test_compaction_keeps_page_text_and_shrinks_input():
    _, base_seen = _run(agent_module.agent)
    result, compact_seen = _run(agent_module.tree_compact_agent)

    # 同じ回数だけモデルが呼ばれる（順路は台本で固定）
    assert len(base_seen) == len(compact_seen) == len(ROUTE) + 1

    # 最後のリクエストでも、根拠ページの本文はすべて残っている
    final_input = compact_seen[-1]
    for page_id in PAGE_IDS:
        page = json.loads(browse_tree(page_id))["node"]
        assert json.dumps(page["content"], ensure_ascii=False)[1:-1] in final_input

    # 目次の候補一覧は短い記録に置き換わり、IDという参照は残る
    assert "確認済み" in final_input
    assert '"lec03-main-s005"' in final_input
    # v2: 章の境界（子のID一覧）は残る。Part 6 は p041 で終わり、p042 は含まれない
    assert '"child_ids": ["lec03-main-p037", "lec03-main-p038", "lec03-main-p039", "lec03-main-p040", "lec03-main-p041"]' in final_input

    # モデルに届いたツール結果の合計（全リクエスト分）が減る
    base_total = sum(map(len, base_seen))
    compact_total = sum(map(len, compact_seen))
    print(f"\nツール結果の累計文字数: tree={base_total:,} / tree-compact={compact_total:,} "
          f"({compact_total / base_total:.0%})")
    assert compact_total < base_total

    # 元の目次は metadata に退避されている（モデルには送られない）
    stashed = [p for m in result.all_messages() for p in m.parts
               if isinstance(p, ToolReturnPart) and COMPACTED_KEY in (p.metadata or {})]
    assert stashed and all(json.loads(p.metadata[COMPACTED_KEY])["children"] for p in stashed)


def test_trace_shows_original_toc(capsys):
    result, _ = _run(agent_module.tree_compact_agent)
    agent_module.print_trace(result)
    out = capsys.readouterr().out
    assert "短い記録に置き換え済み" in out
    assert "lec03-main-p039" in out  # 置き換えた目次の候補も trace では読める
