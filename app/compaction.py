"""第5回 宿題（発展）: Tree の目次探索に「軽いコンパクション」を足す。

ねらい（本講義6章）:
- 必要なときに取りに行く（ジャストインタイム）: 目次の中身は捨てても、ノードIDという「参照」は残す。
  必要になれば同じIDで browse_tree を呼び直せばよい。
- コンパクション：まず残す: 回答の根拠になる「ページ本文」は消さない。
  消すのは、たどり終えた「目次の候補一覧」だけ（最も軽い手＝古いツール結果の消去）。

元の内容は ToolReturnPart.metadata に退避する。metadata はモデルには送られないので、
--trace の表示では元の目次を確認できる。
"""

import json
from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart

COMPACTED_KEY = "compacted_original"


def _is_candidate_list(part: ToolReturnPart) -> bool:
    """browse_tree の戻り値のうち、候補一覧（children あり）なら True。ページ本文は False。"""
    if part.tool_name != "browse_tree" or not isinstance(part.content, str):
        return False
    try:
        data = json.loads(part.content)
    except json.JSONDecodeError:
        return False
    return bool(data.get("children"))


def _stub(part: ToolReturnPart) -> str:
    """候補一覧を、ノードIDと見出し、子のID一覧だけの短い記録に置き換える。

    v2: 子のID一覧は残す。v1 では一覧ごと消したため、章の終わりが分からなくなり、
    モデルが連番から存在しないはずの候補IDを推測して開いた（第3回 p.041 の次に p.042）。
    先頭文は捨てても、「この章にはどのページがあるか」という境界の情報は残す。
    """
    data = json.loads(part.content)
    node = data["node"]
    return json.dumps(
        {
            "node": {key: node[key] for key in ("id", "kind", "title")},
            "children": [],
            "child_ids": [child["id"] for child in data["children"]],
            "compacted": "この目次は確認済み。child_ids がこの目次の候補のすべて。"
                         "先頭文をもう一度見るときは同じIDで browse_tree を呼ぶ。",
        },
        ensure_ascii=False,
    )


def compact_tree_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """モデルへ送る直前に、古い目次の候補一覧を短い記録に置き換える。

    - 最新のリクエスト（モデルがまだ読んでいない結果）は触らない
    - ページ本文（children が空の戻り値）は残す
    """
    last_request = max(
        (i for i, m in enumerate(messages) if isinstance(m, ModelRequest)), default=-1
    )
    compacted = []
    for i, message in enumerate(messages):
        if not isinstance(message, ModelRequest) or i == last_request:
            compacted.append(message)
            continue
        parts = []
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and _is_candidate_list(part):
                metadata = {**(part.metadata or {}), COMPACTED_KEY: part.content}
                part = replace(part, content=_stub(part), metadata=metadata)
            parts.append(part)
        compacted.append(replace(message, parts=parts))
    return compacted
