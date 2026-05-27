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
SCHEMA_TYPES = ("thesis", "principle", "decision", "constraint", "question", "deprecated")
DEFAULT_RELATION_TYPES = ("supports", "conflicts", "depends_on", "supersedes", "implements")
SCHEMA_TYPE_LABELS = {
    "thesis": "thesis 核心判断",
    "principle": "principle 原则",
    "decision": "decision 决策",
    "constraint": "constraint 约束",
    "question": "question 问题",
    "deprecated": "deprecated 废弃",
}
RELATION_TYPE_LABELS = {
    "supports": "supports 支持",
    "conflicts": "conflicts 冲突",
    "depends_on": "depends_on 依赖",
    "supersedes": "supersedes 替代",
    "implements": "implements 实现",
    "requires": "requires 需要",
    "constrains": "constrains 约束",
    "constrained_by": "constrained_by 受约束",
    "enables": "enables 使能",
    "enabled_by": "enabled_by 被使能",
    "implies": "implies 推导",
    "refines": "refines 细化",
    "raises": "raises 引出",
    "replaces": "replaces 替换",
    "extends": "extends 扩展",
    "implemented_by": "implemented_by 被实现",
    "drives": "drives 驱动",
    "feeds": "feeds 输入",
    "precedes": "precedes 前置",
    "explained_by": "explained_by 被解释",
    "opens": "opens 开启",
    "produces": "produces 产出",
    "separate_from": "separate_from 分离",
    "strengthened_by": "strengthened_by 被强化",
    "preserves": "preserves 保留",
    "responds_to": "responds_to 响应",
    "guides": "guides 指导",
    "contains": "contains 包含",
    "satisfies": "satisfies 满足",
    "inspires": "inspires 启发",
    "superseded_by": "superseded_by 被替代",
    "reallocates_priority_to": "reallocates_priority_to 重新分配优先级",
    "informs": "informs 提供信息",
    "organizes": "organizes 组织",
    "resolves": "resolves 解决",
    "validates": "validates 验证",
    "questions": "questions 质疑",
}
ALLOWED_STATUSES = {"active", "draft", "resolved", "deprecated", "conflict"}
ALLOWED_LANGUAGES = {"auto", "zh", "en"}
INDEX_LAYERS = {
    "source": {
        "en": "source 溯源索引",
        "zh": "source 溯源索引",
    },
    "schema": {
        "en": "schema 判断模型索引",
        "zh": "schema 判断模型索引",
    },
    "relations": {
        "en": "relations 关系索引",
        "zh": "relations 关系索引",
    },
    "runtime": {
        "en": "runtime 交接索引",
        "zh": "runtime 交接索引",
    },
}
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


def yaml_quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def clean_wikilink_label(value: Any) -> str:
    return re.sub(r"[\[\]\|\n\r]+", " ", str(value)).strip() or "Untitled"


def graph_neutral_text(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    return text.replace("[[", "[ [").replace("]]", "] ]").strip()


def markdown_list(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["-"]


def markdown_link(path: str, label: str) -> str:
    stem = str(Path(path).with_suffix(""))
    return f"[[{stem}|{clean_wikilink_label(label)}]]"


def project_root_rel(state: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / str(state["project"]["id"])


def project_path_rel(state: dict[str, Any], rel_path: str | Path) -> str:
    return str(project_root_rel(state) / Path(rel_path))


def project_index_filename(state: dict[str, Any]) -> str:
    return f"{slugify(str(state['project']['id']), 'project')}.md"


def project_log_filename(suffix: str = "md") -> str:
    return f"log 时间线.{suffix}"


def layer_filename(state: dict[str, Any], layer: str, suffix: str) -> str:
    return f"{safe_filename_stem(index_title(layer, 'zh'))}.{suffix}"


def schema_type_filename(node_type: str, suffix: str) -> str:
    return f"{safe_filename_stem(schema_type_label(node_type))}.{suffix}"


def relation_type_slug(relation_type: str) -> str:
    value = str(relation_type or "relates_to").strip().lower()
    value = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "-", value)
    value = value.strip("-_")
    return value or "relates_to"


def relation_type_filename(relation_type: str, suffix: str) -> str:
    label = relation_type_label(relation_type)
    if label == str(relation_type or "relates_to"):
        label = relation_type_slug(relation_type)
    return f"{safe_filename_stem(label)}.{suffix}"


def schema_type_label(node_type: str) -> str:
    return SCHEMA_TYPE_LABELS.get(str(node_type), str(node_type))


def relation_type_label(relation_type: str) -> str:
    relation_type = str(relation_type or "relates_to")
    return RELATION_TYPE_LABELS.get(relation_type, relation_type)


def safe_filename_stem(value: str) -> str:
    value = re.sub(r"[/\\:*?\"<>|\n\r]+", "-", str(value)).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(". ")
    return value or "index"


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
        "indexes": {},
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


def node_filename(node: dict[str, Any]) -> str:
    title = str(node.get("title") or node.get("id") or "node")
    stem = slugify(title, "node")[:90].strip("-") or "node"
    return f"{stem}-{short_hash(str(node.get('id') or title), 6)}.md"


def node_link(node: dict[str, Any]) -> str:
    obsidian = node.get("obsidian") if isinstance(node.get("obsidian"), dict) else {}
    path = str(obsidian.get("vaultPath") or obsidian.get("path") or f"nodes/{node_filename(node)}")
    return markdown_link(path, str(node.get("title") or node.get("id") or "Untitled"))


def entry_link(entry_id: str, label: str | None = None, state: dict[str, Any] | None = None) -> str:
    path = f"entries/{entry_id}.md"
    if state is not None:
        path = project_path_rel(state, path)
    return markdown_link(path, label or entry_id)


def index_link(name: str, language: str, state: dict[str, Any] | None = None) -> str:
    title = INDEX_LAYERS.get(name, {}).get(language) or INDEX_LAYERS.get(name, {}).get("en") or name
    if state is not None:
        return markdown_link(project_path_rel(state, f"indexes/{layer_filename(state, name, 'md')}"), title)
    return markdown_link(f"indexes/{name}.md", title)


def project_index_link(state: dict[str, Any], language: str) -> str:
    title = f"{state['project']['name']} / {'Project Index' if language == 'en' else '项目索引'}"
    return markdown_link(project_path_rel(state, project_index_filename(state)), title)


def project_log_link(state: dict[str, Any]) -> str:
    return markdown_link(project_path_rel(state, project_log_filename()), "log 时间线")


def schema_type_index_link(node_type: str, state: dict[str, Any]) -> str:
    return markdown_link(project_path_rel(state, Path("indexes") / "schema" / schema_type_filename(node_type, "md")), schema_type_label(node_type))


def relation_type_index_link(relation_type: str, state: dict[str, Any]) -> str:
    return markdown_link(
        project_path_rel(state, Path("indexes") / "relations" / relation_type_filename(relation_type, "md")),
        relation_type_label(relation_type),
    )


def relation_types_for_state(state: dict[str, Any]) -> list[str]:
    seen = {str(edge.get("type") or "relates_to") for edge in state.get("edges", [])}
    ordered: list[str] = []
    for relation_type in DEFAULT_RELATION_TYPES:
        if relation_type in seen:
            ordered.append(relation_type)
            seen.remove(relation_type)
    ordered.extend(sorted(seen))
    return ordered


def attach_obsidian_metadata(state: dict[str, Any], entry: dict[str, Any]) -> None:
    for node in state.get("nodes", []):
        obsidian = node.setdefault("obsidian", {})
        obsidian.setdefault("path", f"nodes/{node_filename(node)}")
        obsidian["vaultPath"] = project_path_rel(state, str(obsidian["path"]))
        obsidian["link"] = node_link(node)
    entry["obsidian"] = {
        "path": f"entries/{entry['id']}.md",
        "vaultPath": project_path_rel(state, f"entries/{entry['id']}.md"),
        "link": entry_link(entry["id"], entry.get("createdAt"), state),
    }


def frontmatter_lines(values: dict[str, Any]) -> list[str]:
    lines = ["---"]
    for key, value in values.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {yaml_quote(item)}")
        elif isinstance(value, dict):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {yaml_quote(value)}")
    lines.append("---")
    return lines


def edge_lists_for_node(state: dict[str, Any], node_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outgoing = [edge for edge in state.get("edges", []) if edge.get("from") == node_id]
    incoming = [edge for edge in state.get("edges", []) if edge.get("to") == node_id]
    return outgoing, incoming


def node_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("id")): node for node in state.get("nodes", [])}


def render_relation_lines(edges: list[dict[str, Any]], nodes_by_id: dict[str, dict[str, Any]], direction: str) -> list[str]:
    lines: list[str] = []
    for edge in edges:
        other_id = edge.get("to") if direction == "outgoing" else edge.get("from")
        other = nodes_by_id.get(str(other_id))
        if not other:
            continue
        label = str(edge.get("label") or edge.get("type") or "relates_to")
        lines.append(f"- {label}: {node_link(other)} (`{edge.get('type', 'relates_to')}`)")
    return lines or ["-"]


def render_node_file(node: dict[str, Any], state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    tags = merge_unique(
        [
            "mind-map/node",
            f"mind-map/type/{node.get('type', 'decision')}",
            f"mind-map/status/{node.get('status', 'active')}",
        ],
        [f"mind-map/{tag}" for tag in listify(node.get("tags"))],
    )
    fm = frontmatter_lines(
        {
            "id": node.get("id", ""),
            "projectId": state["project"]["id"],
            "type": node.get("type", ""),
            "status": node.get("status", ""),
            "priority": node.get("priority", ""),
            "createdAt": node.get("createdAt", ""),
            "updatedAt": node.get("updatedAt", ""),
            "sourceEntries": listify(node.get("sourceEntries")),
            "aliases": [node.get("title", "")],
            "tags": tags,
        }
    )
    lines = [
        *fm,
        "",
        f"# {node.get('title') or labels['untitled']}",
        "",
        str(node.get("summary") or labels["noneRecorded"]),
        "",
        "## Rationale" if language == "en" else "## 理由",
        "",
        str(node.get("rationale") or labels["noneRecorded"]),
        "",
        "## Implications" if language == "en" else "## 影响",
        "",
        *markdown_list(listify(node.get("implications"))),
    ]
    lines.append("")
    return "\n".join(lines)


def write_node_files(project_dir: Path, state: dict[str, Any], language: str) -> None:
    nodes_dir = project_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = set()
    for node in state.get("nodes", []):
        obsidian = node.get("obsidian") if isinstance(node.get("obsidian"), dict) else {}
        rel_path = Path(str(obsidian.get("path") or f"nodes/{node_filename(node)}"))
        if rel_path.parts and rel_path.parts[0] != "nodes":
            rel_path = Path("nodes") / rel_path.name
        expected_paths.add(rel_path)
        (project_dir / rel_path).write_text(render_node_file(node, state, language), encoding="utf-8")


def write_entry(entry_path: Path, payload: dict[str, Any], entry: dict[str, Any], language: str, state: dict[str, Any]) -> None:
    labels = labels_for(language)
    nodes_by_id = node_by_id(state)
    node_links = [
        node_link(nodes_by_id[node_id])
        for node_id in entry.get("nodeIds", [])
        if node_id in nodes_by_id
    ]
    lines = [
        *frontmatter_lines(
            {
                "id": entry["id"],
                "createdAt": entry["createdAt"],
                "projectId": entry["projectId"],
                "language": language,
                "tags": ["mind-map/entry"],
            }
        ),
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
    lines.extend([f"- {link}" for link in node_links] or ["-"])
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


def index_title(name: str, language: str) -> str:
    return INDEX_LAYERS.get(name, {}).get(language) or INDEX_LAYERS.get(name, {}).get("en") or name


def render_source_index(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    title = index_title("source", language)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "source",
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [title],
                "tags": ["mind-map/index", "mind-map/index/source"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        "",
        f"## {labels['sourceEntries']}",
        "",
    ]
    for entry in state.get("entries", []):
        entry_id = graph_neutral_text(entry.get("id", ""))
        created_at = graph_neutral_text(entry.get("createdAt", ""))
        summary = graph_neutral_text(entry.get("summary", ""))
        lines.append(f"- `{entry_id}` - `{created_at}` - {summary} - path: `entries/{entry_id}.md`")
    lines.extend(["", f"## {labels['nodes']}", ""])
    for node in state.get("nodes", []):
        node_id = graph_neutral_text(node.get("id", ""))
        title_text = graph_neutral_text(node.get("title") or labels["untitled"])
        node_type = graph_neutral_text(node.get("type", ""))
        status = graph_neutral_text(node.get("status", ""))
        sources = ", ".join(f"`{graph_neutral_text(entry_id)}`" for entry_id in listify(node.get("sourceEntries"))) or "-"
        obsidian = node.get("obsidian") if isinstance(node.get("obsidian"), dict) else {}
        node_path = graph_neutral_text(obsidian.get("path") or f"nodes/{node_filename(node)}")
        lines.append(f"- `{node_id}` - {title_text} - `{node_type}` / `{status}` - sourceEntries: {sources} - path: `{node_path}`")
    lines.append("")
    return "\n".join(lines)


def render_schema_index(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    title = index_title("schema", language)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "schema",
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [title],
                "tags": ["mind-map/index", "mind-map/index/schema"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        "",
        "## Type Indexes" if language == "en" else "## 类型索引",
        "",
    ]
    for node_type in SCHEMA_TYPES:
        nodes = [node for node in state.get("nodes", []) if node.get("type") == node_type]
        lines.append(f"- {schema_type_index_link(node_type, state)} - `{len(nodes)}`")
    lines.append("")
    return "\n".join(lines)


def render_schema_type_index(state: dict[str, Any], language: str, node_type: str) -> str:
    labels = labels_for(language)
    nodes = [node for node in state.get("nodes", []) if node.get("type") == node_type]
    display_label = schema_type_label(node_type)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "schema-type",
                "nodeType": node_type,
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [display_label, node_type],
                "tags": ["mind-map/index", "mind-map/index/schema", f"mind-map/type/{node_type}"],
            }
        ),
        "",
        f"# {display_label}",
        "",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        f"- {'Count' if language == 'en' else '数量'}: `{len(nodes)}`",
        "",
        f"## {labels['nodes']}",
        "",
    ]
    for node in nodes:
        lines.append(f"- {node_link(node)} - `{node.get('status', '')}` / `{node.get('priority', '')}`")
    if not nodes:
        lines.append("-")
    lines.append("")
    return "\n".join(lines)


def render_relations_index(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    title = index_title("relations", language)
    relation_types = relation_types_for_state(state)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "relations",
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [title],
                "tags": ["mind-map/index", "mind-map/index/relations"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        "",
        "## Relation Type Indexes" if language == "en" else "## 关系类型索引",
        "",
    ]
    for relation_type in relation_types:
        count = len([edge for edge in state.get("edges", []) if str(edge.get("type") or "relates_to") == relation_type])
        lines.append(f"- {relation_type_index_link(relation_type, state)} - `{count}`")
    if not relation_types:
        lines.append("-")
    lines.append("")
    return "\n".join(lines)


def render_relation_type_index(state: dict[str, Any], language: str, relation_type: str) -> str:
    labels = labels_for(language)
    edges = [edge for edge in state.get("edges", []) if str(edge.get("type") or "relates_to") == relation_type]
    nodes_by_id = node_by_id(state)
    display_label = relation_type_label(relation_type)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "relation-type",
                "relationType": relation_type,
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [display_label, relation_type],
                "tags": ["mind-map/index", "mind-map/index/relations", f"mind-map/relation/{relation_type_slug(relation_type)}"],
            }
        ),
        "",
        f"# {display_label}",
        "",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        f"- {'Count' if language == 'en' else '数量'}: `{len(edges)}`",
        "",
        "## Relations" if language == "en" else "## 关系",
        "",
    ]
    for edge in edges:
        source = nodes_by_id.get(str(edge.get("from")))
        target = nodes_by_id.get(str(edge.get("to")))
        if source and target:
            label = str(edge.get("label") or edge.get("type") or relation_type)
            lines.append(f"- {node_link(source)} -> {node_link(target)} - {label}")
    if not edges:
        lines.append("-")
    lines.append("")
    return "\n".join(lines)


def render_runtime_index(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    context = state.get("context", {})
    task_nodes = [node for node in state.get("nodes", []) if node.get("type") == "task"]
    title = index_title("runtime", language)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "runtime",
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [title],
                "tags": ["mind-map/index", "mind-map/index/runtime"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['projectId']}: `{state['project']['id']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        "",
        f"## {labels['implementationIntent']}",
        "",
        *markdown_list(listify(context.get("implementationIntent"))),
        "",
        "## task nodes",
        "",
    ]
    lines.extend(
        [f"- {node_link(node)} - `{node.get('status', '')}` / `{node.get('priority', '')}`" for node in task_nodes]
        or ["-"]
    )
    lines.extend(["", f"## {labels['acceptanceCriteria']}", ""])
    lines.extend(markdown_list(listify(context.get("acceptanceCriteria"))))
    lines.extend(
        [
            "",
            "## generated artifacts",
            "",
            "- [mindmap.json](../mindmap.json)",
            "- [context.md](../context.md)",
            "- [mindmap.html](../mindmap.html)",
            "",
        ]
    )
    return "\n".join(lines)


def layer_index_data(state: dict[str, Any], layer: str) -> dict[str, Any]:
    base = {
        "schemaVersion": SCHEMA_VERSION,
        "layer": layer,
        "project": state["project"],
        "updatedAt": state["project"]["updatedAt"],
    }
    if layer == "source":
        return {
            **base,
            "entries": state.get("entries", []),
            "nodeSources": [
                {
                    "id": node.get("id"),
                    "title": node.get("title"),
                    "sourceEntries": listify(node.get("sourceEntries")),
                    "obsidian": node.get("obsidian", {}),
                }
                for node in state.get("nodes", [])
            ],
        }
    if layer == "schema":
        return {
            **base,
            "nodes": [
                node
                for node in state.get("nodes", [])
                if node.get("type") in set(SCHEMA_TYPES)
            ],
            "typeIndexes": state.get("indexes", {}).get("schemaTypes", {}),
            "relationsIndex": state.get("indexes", {}).get("relations", {}),
        }
    if layer == "relations":
        return {
            **base,
            "edges": state.get("edges", []),
            "typeIndexes": state.get("indexes", {}).get("relationTypes", {}),
        }
    return {
        **base,
        "implementationIntent": listify(state.get("context", {}).get("implementationIntent")),
        "acceptanceCriteria": listify(state.get("context", {}).get("acceptanceCriteria")),
        "tasks": [node for node in state.get("nodes", []) if node.get("type") == "task"],
    }


def schema_type_index_data(state: dict[str, Any], node_type: str) -> dict[str, Any]:
    nodes = [node for node in state.get("nodes", []) if node.get("type") == node_type]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "layer": "schema-type",
        "nodeType": node_type,
        "project": state["project"],
        "updatedAt": state["project"]["updatedAt"],
        "nodes": nodes,
        "relationsIndex": state.get("indexes", {}).get("relations", {}),
    }


def relation_type_index_data(state: dict[str, Any], relation_type: str) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "layer": "relation-type",
        "relationType": relation_type,
        "project": state["project"],
        "updatedAt": state["project"]["updatedAt"],
        "edges": [
            edge
            for edge in state.get("edges", [])
            if str(edge.get("type") or "relates_to") == relation_type
        ],
    }


def project_log_data(state: dict[str, Any]) -> dict[str, Any]:
    nodes_by_id = node_by_id(state)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "layer": "project-log",
        "project": state["project"],
        "updatedAt": state["project"]["updatedAt"],
        "entries": [
            {
                "id": entry.get("id", ""),
                "createdAt": entry.get("createdAt", ""),
                "summary": entry.get("summary", ""),
                "thought": entry.get("thought", ""),
                "entryPath": f"entries/{entry.get('id', '')}.md",
                "nodeIds": listify(entry.get("nodeIds")),
                "nodes": [
                    {
                        "id": node_id,
                        "title": nodes_by_id.get(str(node_id), {}).get("title", ""),
                        "type": nodes_by_id.get(str(node_id), {}).get("type", ""),
                        "status": nodes_by_id.get(str(node_id), {}).get("status", ""),
                    }
                    for node_id in listify(entry.get("nodeIds"))
                ],
                "edgeIds": listify(entry.get("edgeIds")),
            }
            for entry in state.get("entries", [])
        ],
    }


def render_project_log(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    data = project_log_data(state)
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "project-log",
                "updatedAt": state["project"]["updatedAt"],
                "entryCount": len(data["entries"]),
                "aliases": ["log 时间线", f"{state['project']['name']} log 时间线"],
                "tags": ["mind-map/project-log"],
            }
        ),
        "",
        "# log 时间线",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['projectId']}: `{state['project']['id']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        f"- {'Entry count' if language == 'en' else '记录数'}: `{len(data['entries'])}`",
        "",
        "## Timeline" if language == "en" else "## 时间线",
        "",
    ]
    for entry in data["entries"]:
        entry_id = graph_neutral_text(entry.get("id", ""))
        created_at = graph_neutral_text(entry.get("createdAt", ""))
        summary = graph_neutral_text(entry.get("summary", ""))
        thought = graph_neutral_text(entry.get("thought", ""))
        lines.extend(
            [
                f"### {created_at}",
                "",
                f"- id: `{entry_id}`",
                f"- entryPath: `entries/{entry_id}.md`",
                f"- summary: {summary or '-'}",
            ]
        )
        if thought and thought != summary:
            lines.append(f"- thought: {thought}")
        lines.extend(["", "#### Nodes" if language == "en" else "#### 节点", ""])
        if entry.get("nodes"):
            for node in entry["nodes"]:
                node_id = graph_neutral_text(node.get("id", ""))
                title = graph_neutral_text(node.get("title", ""))
                node_type = graph_neutral_text(node.get("type", ""))
                status = graph_neutral_text(node.get("status", ""))
                lines.append(f"- `{node_id}` - {title} - `{node_type}` / `{status}`")
        else:
            lines.append("-")
        edge_ids = [graph_neutral_text(edge_id) for edge_id in listify(entry.get("edgeIds"))]
        lines.extend(["", "#### Edges" if language == "en" else "#### 关系", ""])
        lines.extend([f"- `{edge_id}`" for edge_id in edge_ids] or ["-"])
        lines.append("")
    if not data["entries"]:
        lines.append("-")
    return "\n".join(lines)


def render_project_index(state: dict[str, Any], language: str) -> str:
    labels = labels_for(language)
    title = f"{state['project']['name']} / {'Project Index' if language == 'en' else '项目索引'}"
    schema_indexes = state.get("indexes", {}).get("schemaTypes", {})
    relation_indexes = state.get("indexes", {}).get("relationTypes", {})
    lines = [
        *frontmatter_lines(
            {
                "projectId": state["project"]["id"],
                "indexLayer": "project",
                "updatedAt": state["project"]["updatedAt"],
                "aliases": [title, state["project"]["name"], state["project"]["id"]],
                "tags": ["mind-map/project-index"],
            }
        ),
        "",
        f"# {title}",
        "",
        f"- {labels['projectPath']}: `{state['project']['path']}`",
        f"- {labels['projectId']}: `{state['project']['id']}`",
        f"- {labels['updated']}: `{state['project']['updatedAt']}`",
        "",
        "## Timeline" if language == "en" else "## 时间线",
        "",
        f"- {project_log_link(state)}",
        "",
        "## Layered Indexes" if language == "en" else "## 分层索引",
        "",
        f"- {index_link('schema', language, state)}",
        f"- {index_link('relations', language, state)}",
        f"- {index_link('runtime', language, state)}",
        "",
        "## All Indexes" if language == "en" else "## 全部索引",
        "",
        "### Schema Types" if language == "en" else "### Schema 类型索引",
        "",
    ]
    for node_type in SCHEMA_TYPES:
        item = schema_indexes.get(node_type, {})
        link = item.get("link") or schema_type_index_link(node_type, state)
        lines.append(f"- {link}")
    lines.extend(
        [
            "",
            "### Relation Types" if language == "en" else "### 关系类型索引",
            "",
        ]
    )
    if relation_indexes:
        for relation_type in sorted(relation_indexes):
            lines.append(f"- {relation_indexes[relation_type].get('link') or relation_type_index_link(relation_type, state)}")
    else:
        lines.append("-")
    lines.extend(
        [
            "",
            "## Generated Artifacts" if language == "en" else "## 生成产物",
            "",
            "- [mindmap.json](mindmap.json)",
            "- [context.md](context.md)",
            "- [mindmap.html](mindmap.html)",
            f"- [{project_log_filename()}]({project_log_filename()})",
            "",
        ]
    )
    return "\n".join(lines)


def write_project_index(project_dir: Path, state: dict[str, Any], language: str) -> None:
    index_path = project_dir / project_index_filename(state)
    index_path.write_text(render_project_index(state, language), encoding="utf-8")
    state["project"]["indexPage"] = {
        "markdown": project_index_filename(state),
        "vaultPath": project_path_rel(state, project_index_filename(state)),
        "link": project_index_link(state, language),
    }


def write_project_log(project_dir: Path, state: dict[str, Any], language: str) -> None:
    md_path = project_dir / project_log_filename("md")
    json_path = project_dir / project_log_filename("json")
    md_path.write_text(render_project_log(state, language), encoding="utf-8")
    json_path.write_text(json.dumps(project_log_data(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["project"]["logPage"] = {
        "markdown": project_log_filename("md"),
        "json": project_log_filename("json"),
        "vaultPath": project_path_rel(state, project_log_filename("md")),
        "link": project_log_link(state),
    }


def write_layer_indexes(project_dir: Path, state: dict[str, Any], language: str) -> None:
    indexes_dir = project_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    renderers = {
        "source": render_source_index,
        "schema": render_schema_index,
        "relations": render_relations_index,
        "runtime": render_runtime_index,
    }
    state["indexes"] = {}
    expected_root_files: set[Path] = set()
    for layer, renderer in renderers.items():
        for legacy_path in (
            indexes_dir / f"{state['project']['id']}-{layer}.md",
            indexes_dir / f"{state['project']['id']}-{layer}.json",
            indexes_dir / f"{layer}.md",
            indexes_dir / f"{layer}.json",
        ):
            if legacy_path.exists():
                legacy_path.unlink()
        md_name = layer_filename(state, layer, "md")
        json_name = layer_filename(state, layer, "json")
        md_path = indexes_dir / md_name
        json_path = indexes_dir / json_name
        expected_root_files.update({md_path, json_path})
        md_path.write_text(renderer(state, language), encoding="utf-8")
        json_path.write_text(json.dumps(layer_index_data(state, layer), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state["indexes"][layer] = {
            "markdown": str(Path("indexes") / md_name),
            "json": str(Path("indexes") / json_name),
            "vaultPath": project_path_rel(state, Path("indexes") / md_name),
            "link": index_link(layer, language, state),
        }
    schema_dir = indexes_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_type_indexes: dict[str, dict[str, str]] = {}
    expected_schema_files: set[Path] = set()
    for node_type in SCHEMA_TYPES:
        md_name = schema_type_filename(node_type, "md")
        json_name = schema_type_filename(node_type, "json")
        md_path = schema_dir / md_name
        json_path = schema_dir / json_name
        expected_schema_files.update({md_path, json_path})
        md_path.write_text(render_schema_type_index(state, language, node_type), encoding="utf-8")
        json_path.write_text(json.dumps(schema_type_index_data(state, node_type), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        schema_type_indexes[node_type] = {
            "markdown": str(Path("indexes") / "schema" / md_name),
            "json": str(Path("indexes") / "schema" / json_name),
            "vaultPath": project_path_rel(state, Path("indexes") / "schema" / md_name),
            "link": schema_type_index_link(node_type, state),
        }
    for stale_path in schema_dir.glob("*"):
        if stale_path.is_file() and stale_path.suffix in {".md", ".json"} and stale_path not in expected_schema_files:
            stale_path.unlink()
    state["indexes"]["schemaTypes"] = schema_type_indexes
    relations_dir = indexes_dir / "relations"
    relations_dir.mkdir(parents=True, exist_ok=True)
    relation_type_indexes: dict[str, dict[str, str]] = {}
    expected_relation_files: set[Path] = set()
    for relation_type in relation_types_for_state(state):
        md_name = relation_type_filename(relation_type, "md")
        json_name = relation_type_filename(relation_type, "json")
        md_path = relations_dir / md_name
        json_path = relations_dir / json_name
        expected_relation_files.update({md_path, json_path})
        md_path.write_text(render_relation_type_index(state, language, relation_type), encoding="utf-8")
        json_path.write_text(json.dumps(relation_type_index_data(state, relation_type), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        relation_type_indexes[relation_type] = {
            "markdown": str(Path("indexes") / "relations" / md_name),
            "json": str(Path("indexes") / "relations" / json_name),
            "vaultPath": project_path_rel(state, Path("indexes") / "relations" / md_name),
            "link": relation_type_index_link(relation_type, state),
        }
    for stale_path in relations_dir.glob("*"):
        if stale_path.is_file() and stale_path.suffix in {".md", ".json"} and stale_path not in expected_relation_files:
            stale_path.unlink()
    state["indexes"]["relationTypes"] = relation_type_indexes
    # Re-render layer pages after child index metadata exists.
    schema_md = indexes_dir / layer_filename(state, "schema", "md")
    schema_json = indexes_dir / layer_filename(state, "schema", "json")
    relations_md = indexes_dir / layer_filename(state, "relations", "md")
    relations_json = indexes_dir / layer_filename(state, "relations", "json")
    schema_md.write_text(render_schema_index(state, language), encoding="utf-8")
    schema_json.write_text(json.dumps(layer_index_data(state, "schema"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    relations_md.write_text(render_relations_index(state, language), encoding="utf-8")
    relations_json.write_text(json.dumps(layer_index_data(state, "relations"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for stale_path in indexes_dir.glob("*"):
        if stale_path.is_file() and stale_path.suffix in {".md", ".json"} and stale_path not in expected_root_files:
            stale_path.unlink()


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


def project_summary_item(state: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    nodes = state.get("nodes", [])
    entries = state.get("entries", [])
    latest_entry = entries[-1] if entries else {}
    active_statuses = {"active", "draft", "conflict"}
    active_nodes = [node for node in nodes if node.get("status") in active_statuses]
    active_tasks = [node for node in nodes if node.get("type") == "task" and node.get("status") in active_statuses]
    open_questions = [node for node in nodes if node.get("type") == "question" and node.get("status") in active_statuses]
    active_decisions = [node for node in nodes if node.get("type") == "decision" and node.get("status") in active_statuses]
    return {
        "id": state["project"]["id"],
        "name": state["project"]["name"],
        "path": state["project"]["path"],
        "updatedAt": state["project"]["updatedAt"],
        "mindmap": str(project_dir / "mindmap.html"),
        "context": str(project_dir / "context.md"),
        "log": str(project_dir / project_log_filename("md")),
        "projectIndex": str(project_dir / project_index_filename(state)),
        "indexes": str(project_dir / "indexes"),
        "nodeCount": len(nodes),
        "edgeCount": len(state.get("edges", [])),
        "entryCount": len(entries),
        "activeNodeCount": len(active_nodes),
        "activeDecisionCount": len(active_decisions),
        "activeTaskCount": len(active_tasks),
        "openQuestionCount": len(open_questions),
        "latestEntry": {
            "id": latest_entry.get("id", ""),
            "createdAt": latest_entry.get("createdAt", ""),
            "summary": latest_entry.get("summary", ""),
        },
        "links": {
            "projectIndex": project_index_link(state, state.get("settings", {}).get("language", "zh")),
            "log": project_log_link(state),
            "source": index_link("source", state.get("settings", {}).get("language", "zh"), state),
            "schema": index_link("schema", state.get("settings", {}).get("language", "zh"), state),
            "relations": index_link("relations", state.get("settings", {}).get("language", "zh"), state),
            "runtime": index_link("runtime", state.get("settings", {}).get("language", "zh"), state),
        },
    }


def state_path_from_project_item(item: dict[str, Any]) -> Path | None:
    for key in ("mindmap", "projectIndex", "context"):
        value = item.get(key)
        if value:
            return Path(str(value)).expanduser().parent / "mindmap.json"
    project_id = item.get("id")
    if project_id:
        return Path(str(project_id)) / "mindmap.json"
    return None


def refresh_project_items(projects: list[dict[str, Any]], current_item: dict[str, Any]) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    by_id = {str(item.get("id")): item for item in projects if item.get("id")}
    by_id[str(current_item["id"])] = current_item
    for item in by_id.values():
        state_path = state_path_from_project_item(item)
        if state_path and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                refreshed.append(project_summary_item(state, state_path.parent))
                continue
            except Exception:
                pass
        refreshed.append(item)
    refreshed.sort(key=lambda item: str(item.get("updatedAt", "")), reverse=True)
    return refreshed


def truncate_cell(value: Any, length: int = 120) -> str:
    text = " ".join(str(value or "").replace("|", "/").split())
    if len(text) <= length:
        return text
    return text[: length - 1] + "..."


def render_vault_project_index(index: dict[str, Any], root: Path) -> str:
    updated_at = index.get("updatedAt") or now_iso()
    projects = index.get("projects", [])
    lines = [
        *frontmatter_lines(
            {
                "indexLayer": "all-projects",
                "updatedAt": updated_at,
                "projectCount": len(projects),
                "tags": ["mind-map/all-projects-index"],
                "aliases": ["MindMaps 项目总索引", "MindMaps Projects"],
            }
        ),
        "",
        "# MindMaps 项目总索引",
        "",
        f"- 更新时间: `{updated_at}`",
        f"- 项目数: `{len(projects)}`",
        "",
        "## 项目状态",
        "",
        "| 项目 | 最新状态 | 更新时间 | 节点 | 任务 | 问题 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in projects:
        project_id = str(item.get("id", ""))
        name = str(item.get("name") or project_id)
        project_link = markdown_link(str(OUTPUT_ROOT / project_id / project_index_filename({"project": {"id": project_id}})), name)
        latest = truncate_cell(item.get("latestEntry", {}).get("summary") or item.get("path") or "")
        updated = str(item.get("updatedAt", ""))
        node_count = item.get("nodeCount", "")
        task_count = item.get("activeTaskCount", "")
        question_count = item.get("openQuestionCount", "")
        lines.append(
            f"| {project_link} | {latest} | `{updated}` | {node_count} | {task_count} | {question_count} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 总索引只链接项目目录页。",
            "- 项目目录页链接本项目的可视索引，用作项目图谱锚点。",
            "- source 作为后台溯源层保留，不挂到项目目录页。",
            "- 子索引页不再反向链接上级索引。",
            "",
        ]
    )
    return "\n".join(lines)


def update_index(index_path: Path, state: dict[str, Any], project_dir: Path) -> None:
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schemaVersion": SCHEMA_VERSION, "projects": []}
    index["updatedAt"] = now_iso()
    index["projects"] = refresh_project_items(
        index.get("projects", []),
        project_summary_item(state, project_dir),
    )
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_path.parent / "index.md").write_text(render_vault_project_index(index, index_path.parent), encoding="utf-8")


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
    attach_obsidian_metadata(state, entry)

    context = state.setdefault("context", {})
    context["implementationIntent"] = merge_unique(
        listify(context.get("implementationIntent")), listify(payload.get("implementationIntent"))
    )
    context["acceptanceCriteria"] = merge_unique(
        listify(context.get("acceptanceCriteria")), listify(payload.get("acceptanceCriteria"))
    )

    write_entry(entries_dir / f"{entry_id}.md", payload, entry, language, state)
    write_node_files(project_dir, state, language)
    write_layer_indexes(project_dir, state, language)
    write_project_log(project_dir, state, language)
    write_project_index(project_dir, state, language)
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
            "log": str(project_dir / project_log_filename("md")),
            "state": str(state_path),
            "entry": str(entries_dir / f"{entry_id}.md"),
            "projectIndex": str(project_dir / project_index_filename(state)),
            "nodesDir": str(project_dir / "nodes"),
            "indexesDir": str(project_dir / "indexes"),
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
