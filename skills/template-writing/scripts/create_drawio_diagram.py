#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DEFAULT_NODE_STYLE = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;"
    "strokeColor=#6c8ebf;fontSize=12;spacing=8;"
)
DEFAULT_EDGE_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;"
)
DEFAULT_MARGIN_X = 80
DEFAULT_MARGIN_Y = 90
DEFAULT_NODE_WIDTH = 170
DEFAULT_NODE_HEIGHT = 64
DEFAULT_H_GAP = 70
DEFAULT_V_GAP = 56
DEFAULT_MAX_WIDTH = 900


def safe_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def node_positions(
    count: int,
    layout: str,
    columns: int,
    node_width: int,
    node_height: int,
    h_gap: int,
    v_gap: int,
    margin_x: int,
    margin_y: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    columns = max(1, columns)
    for index in range(count):
        if layout == "vertical":
            x = margin_x + (index // 6) * (node_width + h_gap)
            y = margin_y + (index % 6) * (node_height + v_gap)
        elif layout in {"grid", "horizontal"}:
            x = margin_x + (index % columns) * (node_width + h_gap)
            y = margin_y + (index // columns) * (node_height + v_gap)
        else:
            x = margin_x + index * (node_width + h_gap)
            y = margin_y
        positions.append((x, y))
    return positions


def layout_columns(spec: dict[str, Any], count: int, node_width: int, h_gap: int, margin_x: int) -> int:
    if "columns" in spec:
        return max(1, int(spec["columns"]))
    max_width = int(spec.get("max_width", DEFAULT_MAX_WIDTH))
    usable = max(node_width, max_width - (2 * margin_x))
    return max(1, min(count, (usable + h_gap) // (node_width + h_gap)))


def diagram_size(
    positions: list[tuple[int, int]],
    node_width: int,
    node_height: int,
    margin_x: int,
    margin_y: int,
) -> tuple[int, int]:
    if not positions:
        return 1169, 827
    max_x = max(x for x, _ in positions) + node_width + margin_x
    max_y = max(y for _, y in positions) + node_height + margin_y
    return max(700, max_x), max(420, max_y)


def add_node(parent: ET.Element, node: dict[str, Any], node_id: str, x: int, y: int) -> None:
    width = int(node.get("width", 170))
    height = int(node.get("height", 64))
    cell = ET.SubElement(
        parent,
        "mxCell",
        {
            "id": node_id,
            "value": str(node.get("label", node_id)),
            "style": str(node.get("style", DEFAULT_NODE_STYLE)),
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": str(int(node.get("x", x))),
            "y": str(int(node.get("y", y))),
            "width": str(width),
            "height": str(height),
            "as": "geometry",
        },
    )


def add_edge(parent: ET.Element, edge: dict[str, Any], edge_id: str, source: str, target: str) -> None:
    attrs = {
        "id": edge_id,
        "value": str(edge.get("label", "")),
        "style": str(edge.get("style", DEFAULT_EDGE_STYLE)),
        "edge": "1",
        "parent": "1",
        "source": source,
        "target": target,
    }
    cell = ET.SubElement(parent, "mxCell", attrs)
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


def build_drawio(spec: dict[str, Any]) -> ET.Element:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "template-writing",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "diagram-1", "name": str(spec.get("title", "Page-1"))})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1422",
            "dy": "794",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1169",
            "pageHeight": "827",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    nodes = list(spec.get("nodes", []))
    if not nodes:
        raise ValueError("diagram spec must contain at least one node")

    layout = str(spec.get("layout", "horizontal"))
    node_width = int(spec.get("node_width", DEFAULT_NODE_WIDTH))
    node_height = int(spec.get("node_height", DEFAULT_NODE_HEIGHT))
    h_gap = int(spec.get("h_gap", DEFAULT_H_GAP))
    v_gap = int(spec.get("v_gap", DEFAULT_V_GAP))
    margin_x = int(spec.get("margin_x", DEFAULT_MARGIN_X))
    margin_y = int(spec.get("margin_y", DEFAULT_MARGIN_Y))
    columns = layout_columns(spec, len(nodes), node_width, h_gap, margin_x)
    positions = node_positions(len(nodes), layout, columns, node_width, node_height, h_gap, v_gap, margin_x, margin_y)
    page_width, page_height = diagram_size(positions, node_width, node_height, margin_x, margin_y)
    model.set("pageWidth", str(page_width))
    model.set("pageHeight", str(page_height))
    id_map: dict[str, str] = {}

    for index, node in enumerate(nodes):
        original_id = str(node.get("id", f"node_{index + 1}"))
        node_id = safe_id(original_id, f"node_{index + 1}")
        id_map[original_id] = node_id
        add_node(root, node, node_id, *positions[index])

    for index, edge in enumerate(spec.get("edges", [])):
        source_key = str(edge["source"])
        target_key = str(edge["target"])
        if source_key not in id_map or target_key not in id_map:
            raise ValueError(f"edge references unknown node: {source_key!r} -> {target_key!r}")
        edge_id = safe_id(str(edge.get("id", f"edge_{index + 1}")), f"edge_{index + 1}")
        add_edge(root, edge, edge_id, id_map[source_key], id_map[target_key])

    return mxfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an editable draw.io diagram from a JSON spec.")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    tree = ET.ElementTree(build_drawio(spec))
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
