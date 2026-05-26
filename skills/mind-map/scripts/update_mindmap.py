#!/usr/bin/env python3
"""Update an Obsidian-hosted project mind map.

The script is intentionally deterministic and dependency-free. The agent using
the skill performs semantic extraction; this script handles durable storage,
merge hygiene, HTML rendering, and AI context-pack generation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mind-map.v1"
DEFAULT_VAULT = Path("/Users/echooo/SynologyDrive/Typora")
OUTPUT_ROOT = Path("AI-Projects/MindMaps")
ALLOWED_TYPES = {
    "thesis",
    "principle",
    "decision",
    "constraint",
    "question",
    "task",
    "deprecated",
}
ALLOWED_STATUSES = {"active", "draft", "resolved", "deprecated", "conflict"}
ALLOWED_LANGUAGES = {"auto", "zh", "en"}
LABELS = {
    "en": {
        "pageTitle": "Mind Map",
        "contextPack": "Project Context Pack",
        "meta": "Project mind map · Updated {updatedAt} · Read context.md before implementation",
        "futureNote": "Future AI agents should read this file before implementing changes for this project.",
        "projectPath": "Project path",
        "projectId": "Project id",
        "updated": "Updated",
        "productThesis": "Product Thesis",
        "currentDirection": "Current Direction",
        "nonNegotiables": "Non-Negotiables",
        "activeDecisions": "Active Decisions",
        "openQuestions": "Open Questions",
        "implementationIntent": "Implementation Intent",
        "deprecatedIdeas": "Deprecated Ideas",
        "acceptanceCriteria": "Acceptance Criteria",
        "sourceEntries": "Source Entries",
        "noneRecorded": "None recorded.",
        "noThesis": "No explicit thesis recorded yet.",
        "rationale": "Rationale",
        "status": "Status",
        "priority": "Priority",
        "entryTitle": "Mind Map Entry",
        "userThought": "User Thought",
        "structuredSummary": "Structured Summary",
        "nodes": "Nodes",
        "rawPayload": "Raw Payload",
        "untitled": "Untitled",
    },
    "zh": {
        "pageTitle": "思维导图",
        "contextPack": "项目上下文包",
        "meta": "项目思维导图 · 更新于 {updatedAt} · 实现前先阅读 context.md",
        "futureNote": "后续 AI 在实现本项目之前，应先阅读这个上下文包。",
        "projectPath": "项目路径",
        "projectId": "项目 ID",
        "updated": "更新时间",
        "productThesis": "项目核心判断",
        "currentDirection": "当前方向",
        "nonNegotiables": "不可违背的原则",
        "activeDecisions": "已确认决策",
        "openQuestions": "未决问题",
        "implementationIntent": "实现意图",
        "deprecatedIdeas": "已废弃方向",
        "acceptanceCriteria": "验收标准",
        "sourceEntries": "来源记录",
        "noneRecorded": "暂无记录。",
        "noThesis": "尚未记录明确的项目核心判断。",
        "rationale": "理由",
        "status": "状态",
        "priority": "优先级",
        "entryTitle": "思维导图记录",
        "userThought": "用户原始想法",
        "structuredSummary": "结构化摘要",
        "nodes": "节点",
        "rawPayload": "原始 Payload",
        "untitled": "未命名",
    },
}


def now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def slugify(value: str, fallback: str = "item") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    value = value.strip("-")
    return value or fallback


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def resolve_vault(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("MIND_MAP_VAULT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    config_path = Path.home() / ".agent-mindmap" / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            configured = data.get("defaultVaultPath")
            if configured:
                return Path(configured).expanduser().resolve()
        except Exception:
            pass
    return DEFAULT_VAULT


def load_config() -> dict[str, Any]:
    config_path = Path.home() / ".agent-mindmap" / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def contains_cjk(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(contains_cjk(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_cjk(item) for item in value)
    return bool(re.search(r"[\u4e00-\u9fff]", str(value)))


def normalize_language(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    return text if text in ALLOWED_LANGUAGES else "auto"


def resolve_language(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    explicit = normalize_language(args.language)
    if explicit != "auto":
        return explicit
    payload_language = normalize_language(payload.get("language"))
    if payload_language != "auto":
        return payload_language
    config_language = normalize_language(load_config().get("defaultLanguage"))
    if config_language != "auto":
        return config_language
    return "zh" if contains_cjk(payload) else "en"


def labels_for(language: str) -> dict[str, str]:
    return LABELS.get(language, LABELS["en"])


def project_identity(project_path: Path) -> dict[str, str]:
    resolved = project_path.expanduser().resolve()
    name = resolved.name or "project"
    project_id = f"{slugify(name, 'project')}-{short_hash(str(resolved), 8)}"
    return {"id": project_id, "name": name, "path": str(resolved)}


def read_payload(args: argparse.Namespace) -> dict[str, Any]:
    raw = ""
    if args.entry_json:
        raw = Path(args.entry_json).expanduser().read_text(encoding="utf-8")
    elif args.thought:
        raw = args.thought
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()

    raw = raw.strip()
    if not raw:
        raise SystemExit("No thought provided. Use --thought, --entry-json, or stdin.")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return {
        "thought": raw,
        "summary": first_sentence(raw),
        "nodes": [heuristic_node(raw)],
        "edges": [],
    }


def first_sentence(text: str) -> str:
    compact = " ".join(text.split())
    match = re.split(r"(?<=[。.!?？])\s+", compact, maxsplit=1)
    return (match[0] if match else compact)[:220]


def heuristic_node(text: str) -> dict[str, Any]:
    lowered = text.lower()
    node_type = "decision"
    if "?" in text or "？" in text or "如何" in text:
        node_type = "question"
    if "原则" in text or "principle" in lowered:
        node_type = "principle"
    if "不要" in text or "不再" in text or "deprecated" in lowered:
        node_type = "deprecated"
    if "下一步" in text or "实现" in text or "task" in lowered:
        node_type = "task"
    return {
        "type": node_type,
        "title": first_sentence(text)[:80],
        "summary": first_sentence(text),
        "rationale": "",
        "status": "active" if node_type != "deprecated" else "deprecated",
        "priority": "medium",
        "implications": [],
        "tags": [],
    }


def load_state(path: Path, project: dict[str, str], vault: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    created = now_iso()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "project": {
            **project,
            "vaultPath": str(vault),
            "createdAt": created,
            "updatedAt": created,
        },
        "nodes": [],
        "edges": [],
        "entries": [],
        "context": {
            "implementationIntent": [],
            "acceptanceCriteria": [],
        },
    }


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in existing + incoming:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(key)
    return merged


def normalize_node(raw: dict[str, Any], entry_id: str, timestamp: str) -> dict[str, Any]:
    title = str(raw.get("title") or raw.get("name") or "Untitled thought").strip()
    node_type = str(raw.get("type") or "decision").strip()
    status = str(raw.get("status") or "active").strip()
    if node_type not in ALLOWED_TYPES:
        node_type = "decision"
    if status not in ALLOWED_STATUSES:
        status = "active"
    node_id = str(raw.get("id") or f"{node_type}-{slugify(title, 'node')}-{short_hash(title, 6)}")
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "summary": str(raw.get("summary") or "").strip(),
        "rationale": str(raw.get("rationale") or "").strip(),
        "status": status,
        "priority": str(raw.get("priority") or "medium").strip(),
        "implications": listify(raw.get("implications")),
        "tags": listify(raw.get("tags")),
        "sourceEntries": [entry_id],
        "createdAt": timestamp,
        "updatedAt": timestamp,
    }


def node_match_key(node: dict[str, Any]) -> tuple[str, str]:
    return (str(node.get("type", "")), slugify(str(node.get("title", ""))))


def merge_node(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key in ["title", "summary", "rationale", "status", "priority", "type"]:
        value = incoming.get(key)
        if value:
            existing[key] = value
    existing["implications"] = merge_unique(
        listify(existing.get("implications")), listify(incoming.get("implications"))
    )
    existing["tags"] = merge_unique(listify(existing.get("tags")), listify(incoming.get("tags")))
    existing["sourceEntries"] = merge_unique(
        listify(existing.get("sourceEntries")), listify(incoming.get("sourceEntries"))
    )
    existing["updatedAt"] = incoming["updatedAt"]
    return existing


def upsert_nodes(state: dict[str, Any], raw_nodes: list[dict[str, Any]], entry_id: str, timestamp: str) -> dict[str, str]:
    title_to_id: dict[str, str] = {}
    by_id = {node["id"]: node for node in state["nodes"]}
    by_key = {node_match_key(node): node for node in state["nodes"]}
    for raw in raw_nodes:
        node = normalize_node(raw, entry_id, timestamp)
        key = node_match_key(node)
        target = by_id.get(node["id"]) or by_key.get(key)
        if target:
            merge_node(target, node)
            title_to_id[node["title"]] = target["id"]
        else:
            state["nodes"].append(node)
            by_id[node["id"]] = node
            by_key[key] = node
            title_to_id[node["title"]] = node["id"]
    return title_to_id


def resolve_node_ref(ref: Any, title_to_id: dict[str, str], state: dict[str, Any]) -> str | None:
    if not ref:
        return None
    text = str(ref)
    if any(node["id"] == text for node in state["nodes"]):
        return text
    if text in title_to_id:
        return title_to_id[text]
    for node in state["nodes"]:
        if node_match_key(node)[1] == slugify(text):
            return node["id"]
    return None


def upsert_edges(state: dict[str, Any], raw_edges: list[dict[str, Any]], title_to_id: dict[str, str], entry_id: str, timestamp: str) -> list[str]:
    edge_ids: list[str] = []
    by_key = {(edge.get("from"), edge.get("to"), edge.get("type")): edge for edge in state["edges"]}
    for raw in raw_edges:
        from_ref = raw.get("from") or raw.get("fromTitle")
        to_ref = raw.get("to") or raw.get("toTitle")
        from_id = resolve_node_ref(from_ref, title_to_id, state)
        to_id = resolve_node_ref(to_ref, title_to_id, state)
        if not from_id or not to_id or from_id == to_id:
            continue
        edge_type = str(raw.get("type") or "relates_to").strip()
        key = (from_id, to_id, edge_type)
        existing = by_key.get(key)
        if existing:
            existing["label"] = str(raw.get("label") or existing.get("label") or edge_type)
            existing["sourceEntries"] = merge_unique(listify(existing.get("sourceEntries")), [entry_id])
            existing["updatedAt"] = timestamp
            edge_ids.append(existing["id"])
        else:
            edge = {
                "id": f"edge-{short_hash('|'.join(key), 10)}",
                "from": from_id,
                "to": to_id,
                "type": edge_type,
                "label": str(raw.get("label") or edge_type),
                "sourceEntries": [entry_id],
                "createdAt": timestamp,
                "updatedAt": timestamp,
            }
            state["edges"].append(edge)
            by_key[key] = edge
            edge_ids.append(edge["id"])
    return edge_ids


def write_entry(entry_path: Path, payload: dict[str, Any], entry: dict[str, Any], language: str) -> None:
    labels = labels_for(language)
    lines = [
        "---",
        f"id: {entry['id']}",
        f"createdAt: {entry['createdAt']}",
        f"projectId: {entry['projectId']}",
        f"language: {language}",
        "---",
        "",
        f"# {labels['entryTitle']} {entry['createdAt']}",
        "",
        f"## {labels['userThought']}",
        "",
        str(payload.get("thought") or "").strip(),
        "",
        f"## {labels['structuredSummary']}",
        "",
        str(payload.get("summary") or "").strip(),
        "",
        f"## {labels['nodes']}",
        "",
    ]
    for node_id in entry.get("nodeIds", []):
        lines.append(f"- `{node_id}`")
    lines.extend(["", f"## {labels['rawPayload']}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""])
    entry_path.write_text("\n".join(lines), encoding="utf-8")


def active_nodes(state: dict[str, Any], *types: str) -> list[dict[str, Any]]:
    return [
        node
        for node in state["nodes"]
        if node.get("type") in types and node.get("status") in {"active", "draft", "conflict"}
    ]


def render_node_md(node: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    lines = [f"### {node.get('title') or labels['untitled']}"]
    if node.get("summary"):
        lines.append(str(node["summary"]))
    if node.get("rationale"):
        lines.extend(["", f"{labels['rationale']}: {node['rationale']}"])
    implications = listify(node.get("implications"))
    if implications:
        lines.append("")
        lines.extend([f"- {item}" for item in implications])
    lines.append("")
    lines.append(f"{labels['status']}: `{node.get('status', 'active')}` | {labels['priority']}: `{node.get('priority', 'medium')}`")
    return "\n".join(lines)


def render_context(state: dict[str, Any]) -> str:
    project = state["project"]
    context = state.get("context", {})
    language = state.get("settings", {}).get("language", "en")
    labels = labels_for(language)
    lines = [
        f"# {project['name']} - {labels['contextPack']}",
        "",
        f"{labels['projectPath']}: `{project['path']}`",
        f"{labels['projectId']}: `{project['id']}`",
        f"{labels['updated']}: `{project['updatedAt']}`",
        "",
        f"> {labels['futureNote']}",
        "",
        f"## {labels['productThesis']}",
        "",
    ]
    thesis_nodes = active_nodes(state, "thesis")
    lines.append("\n\n".join(render_node_md(node, language) for node in thesis_nodes) or labels["noThesis"])
    sections = [
        (labels["currentDirection"], ("decision", "principle")),
        (labels["nonNegotiables"], ("constraint",)),
        (labels["activeDecisions"], ("decision",)),
        (labels["openQuestions"], ("question",)),
        (labels["implementationIntent"], ("task",)),
    ]
    for title, types in sections:
        lines.extend(["", f"## {title}", ""])
        content_nodes = active_nodes(state, *types)
        if title == labels["implementationIntent"]:
            intents = listify(context.get("implementationIntent"))
            if intents:
                lines.extend([f"- {item}" for item in intents])
                lines.append("")
        lines.append("\n\n".join(render_node_md(node, language) for node in content_nodes) or labels["noneRecorded"])

    deprecated = [node for node in state["nodes"] if node.get("type") == "deprecated" or node.get("status") == "deprecated"]
    lines.extend(["", f"## {labels['deprecatedIdeas']}", ""])
    lines.append("\n\n".join(render_node_md(node, language) for node in deprecated) or labels["noneRecorded"])

    criteria = listify(context.get("acceptanceCriteria"))
    lines.extend(["", f"## {labels['acceptanceCriteria']}", ""])
    lines.extend([f"- {item}" for item in criteria] or [labels["noneRecorded"]])

    lines.extend(["", f"## {labels['sourceEntries']}", ""])
    for entry in state.get("entries", [])[-12:]:
        lines.append(f"- `{entry['createdAt']}` - `entries/{entry['id']}.md` - {entry.get('summary', '')}")
    lines.append("")
    return "\n".join(lines)


def html_template() -> str:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "mindmap.html"
    return template_path.read_text(encoding="utf-8")


def render_html(state: dict[str, Any]) -> str:
    data_json = json.dumps(state, ensure_ascii=False)
    safe_data_json = (
        data_json
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    template = html_template()
    language = state.get("settings", {}).get("language", "en")
    labels = labels_for(language)
    return (
        template.replace("__PROJECT_TITLE__", html.escape(state["project"]["name"]))
        .replace("__HTML_LANG__", html.escape(language))
        .replace("__PAGE_TITLE__", html.escape(labels["pageTitle"]))
        .replace("__GENERATED_AT__", html.escape(state["project"]["updatedAt"]))
        .replace("__DATA_JSON__", safe_data_json)
    )


def update_index(index_path: Path, state: dict[str, Any], project_dir: Path) -> None:
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schemaVersion": SCHEMA_VERSION, "projects": []}
    projects = [item for item in index.get("projects", []) if item.get("id") != state["project"]["id"]]
    projects.append(
        {
            "id": state["project"]["id"],
            "name": state["project"]["name"],
            "path": state["project"]["path"],
            "updatedAt": state["project"]["updatedAt"],
            "mindmap": str(project_dir / "mindmap.html"),
            "context": str(project_dir / "context.md"),
        }
    )
    projects.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    index["projects"] = projects
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update project mind map in an Obsidian vault.")
    parser.add_argument("--entry-json", help="Path to a structured entry JSON payload.")
    parser.add_argument("--thought", help="Plain thought text. Used when no structured JSON is available.")
    parser.add_argument("--project-path", default=os.getcwd(), help="Project path. Defaults to cwd.")
    parser.add_argument("--vault", help="Obsidian vault path. Defaults to configured vault.")
    parser.add_argument("--language", choices=sorted(ALLOWED_LANGUAGES), default="auto", help="Output language for generated files.")
    args = parser.parse_args()

    timestamp = now_iso()
    payload = read_payload(args)
    language = resolve_language(args, payload)
    payload.setdefault("language", language)
    vault = resolve_vault(args.vault)
    project = project_identity(Path(args.project_path))
    root = vault / OUTPUT_ROOT
    project_dir = root / project["id"]
    entries_dir = project_dir / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)

    state_path = project_dir / "mindmap.json"
    state = load_state(state_path, project, vault)
    state["project"].update({**project, "vaultPath": str(vault), "updatedAt": timestamp})
    state.setdefault("settings", {})["language"] = language

    entry_id = f"{timestamp.replace(':', '-').replace('+', '-')}-{short_hash(str(payload), 6)}"
    raw_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    if not raw_nodes:
        raw_nodes = [heuristic_node(str(payload.get("thought") or payload.get("summary") or ""))]
    title_to_id = upsert_nodes(state, raw_nodes, entry_id, timestamp)
    edge_ids = upsert_edges(
        state,
        payload.get("edges") if isinstance(payload.get("edges"), list) else [],
        title_to_id,
        entry_id,
        timestamp,
    )

    node_ids = [title_to_id.get(str(node.get("title"))) for node in raw_nodes]
    node_ids = [node_id for node_id in node_ids if node_id]
    entry = {
        "id": entry_id,
        "projectId": project["id"],
        "createdAt": timestamp,
        "projectPath": project["path"],
        "thought": str(payload.get("thought") or "").strip(),
        "summary": str(payload.get("summary") or first_sentence(str(payload.get("thought") or ""))).strip(),
        "language": language,
        "nodeIds": node_ids,
        "edgeIds": edge_ids,
        "source": {"cwd": os.getcwd(), "agent": os.environ.get("USER", "")},
    }
    state["entries"].append(entry)

    context = state.setdefault("context", {})
    context["implementationIntent"] = merge_unique(
        listify(context.get("implementationIntent")), listify(payload.get("implementationIntent"))
    )
    context["acceptanceCriteria"] = merge_unique(
        listify(context.get("acceptanceCriteria")), listify(payload.get("acceptanceCriteria"))
    )

    write_entry(entries_dir / f"{entry_id}.md", payload, entry, language)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (project_dir / "context.md").write_text(render_context(state), encoding="utf-8")
    (project_dir / "mindmap.html").write_text(render_html(state), encoding="utf-8")
    update_index(root / "index.json", state, project_dir)

    print(json.dumps(
        {
            "projectId": project["id"],
            "projectDir": str(project_dir),
            "mindmap": str(project_dir / "mindmap.html"),
            "context": str(project_dir / "context.md"),
            "state": str(state_path),
            "entry": str(entries_dir / f"{entry_id}.md"),
            "language": language,
            "nodesUpdated": len(node_ids),
            "edgesUpdated": len(edge_ids),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
