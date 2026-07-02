from __future__ import annotations

import html
import json
import math
import os
import pickle
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "img"
ART = ROOT / "artifect_all"


PALETTE = {
    "ink": "#172026",
    "muted": "#5f6b73",
    "line": "#c8d1d8",
    "panel": "#f7f9fb",
    "panel2": "#eef4f8",
    "bfs": "#8b98a5",
    "spine": "#0f766e",
    "spine2": "#14b8a6",
    "warn": "#c2410c",
    "ok": "#15803d",
    "gold": "#f59e0b",
    "blue": "#2563eb",
    "red": "#dc2626",
    "purple": "#7c3aed",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def wrap(text: str, width: int) -> List[str]:
    out: List[str] = []
    for part in str(text).split("\n"):
        if not part:
            out.append("")
        else:
            out.extend(textwrap.wrap(part, width=width, break_long_words=False) or [""])
    return out


class SVG:
    def __init__(self, width: int, height: int, title: str = ""):
        self.width = width
        self.height = height
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["muted"]}"/>',
            "</marker>",
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">',
            '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.12"/>',
            "</filter>",
            "</defs>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        ]
        if title:
            self.text(40, 42, title, size=26, weight=700, color=PALETTE["ink"])

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none",
             sw: float = 1, rx: float = 6, shadow: bool = False) -> None:
        filt = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{filt}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#64748b",
             sw: float = 2, arrow: bool = False, dash: str = "") -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{marker}{dash_attr}/>'
        )

    def path(self, d: str, color: str = "#64748b", sw: float = 2, fill: str = "none",
             arrow: bool = False, dash: str = "") -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}" '
            f'stroke-linecap="round" stroke-linejoin="round"{marker}{dash_attr}/>'
        )

    def text(self, x: float, y: float, text: str, size: int = 16, color: str = "#111827",
             weight: int = 400, anchor: str = "start", family: str = "Arial, Helvetica, sans-serif") -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def multiline(self, x: float, y: float, lines: Iterable[str], size: int = 15,
                  color: str = "#111827", weight: int = 400, line_h: int = 20,
                  anchor: str = "start") -> None:
        yy = y
        for line in lines:
            self.text(x, yy, line, size=size, color=color, weight=weight, anchor=anchor)
            yy += line_h

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "white",
               sw: float = 2, shadow: bool = False) -> None:
        filt = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{filt}/>'
        )

    def pill(self, x: float, y: float, text: str, fill: str, color: str = "white",
             size: int = 14, w: float | None = None) -> None:
        if w is None:
            w = max(90, len(text) * 8 + 24)
        self.rect(x, y, w, 30, fill, rx=15)
        self.text(x + w / 2, y + 20, text, size=size, color=color, weight=700, anchor="middle")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def get_metaqa_results() -> Dict[str, Dict[str, Any]]:
    data = load_json(ART / "metaqa-vanilla-test/results/benchmark_results.json")
    out: Dict[str, Dict[str, Any]] = {}
    for name, d in data.items():
        if not isinstance(d, dict):
            continue
        if "Baseline-BFS-gpt-oss" in name:
            out["BFS"] = d
        elif "HRG-Proposed-gpt-oss-json" in name:
            out["HRG JSON"] = d
        elif "HRG-Proposed-gpt-oss-triple" in name:
            out["HRG Triple"] = d
        elif "Spine-Correction-gpt-oss-json" in name:
            out["Spine JSON"] = d
        elif "Spine-Correction-gpt-oss-triple" in name:
            out["Spine Triple"] = d
    return out


def edge_tuple(edge: Any) -> Tuple[str, str, str]:
    if isinstance(edge, dict):
        return str(edge["head"]), str(edge["relation"]), str(edge["tail"])
    return str(edge[0]), str(edge[1]), str(edge[2])


def draw_node(svg: SVG, x: float, y: float, label: str, fill: str, w: int = 128,
              h: int = 42, color: str = "white", stroke: str = "white") -> None:
    svg.rect(x - w / 2, y - h / 2, w, h, fill, stroke=stroke, sw=1.5, rx=8, shadow=True)
    lines = wrap(label.replace("_", " "), 18)
    start_y = y - (len(lines) - 1) * 8 + 5
    for i, line in enumerate(lines[:2]):
        svg.text(x, start_y + i * 16, line, size=13, color=color, weight=700, anchor="middle")


def draw_edge(svg: SVG, x1: float, y1: float, x2: float, y2: float, label: str,
              color: str, sw: float = 2.5, bend: float = 0) -> None:
    if bend:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + bend
        svg.path(f"M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}", color=color, sw=sw, arrow=True)
        lx, ly = mx, my - 5
    else:
        svg.line(x1, y1, x2, y2, color=color, sw=sw, arrow=True)
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 8
    svg.rect(lx - max(48, len(label) * 3.8), ly - 17, max(96, len(label) * 7.6), 24, "white", stroke=color, sw=1, rx=4)
    svg.text(lx, ly, label, size=12, color=color, weight=700, anchor="middle")


def fig_01_pipeline() -> None:
    svg = SVG(1500, 820, "HRG-guided KG-RAG: offline grammar prior and online evidence retrieval")
    svg.multiline(42, 75, wrap("Goal: extract structural priors from the KG, then use them during QA to select compact, executable evidence.", 120), size=18, color=PALETTE["muted"])

    steps = [
        ("Question", "Natural language KGQA question"),
        ("Parse", "LLM outputs top-k entity + relation chain candidates"),
        ("Ground", "Map entity surface form to KG node"),
        ("Validate", "Execute chain over KB; keep only executable paths"),
        ("HRG prior", "Use grammar hit, fallback, and ranking signals"),
        ("Retrieve", "Build compact evidence subgraph"),
        ("Answer", "Serialize evidence and ask LLM"),
    ]
    x0, y, w, h, gap = 70, 185, 175, 115, 30
    centers = []
    for i, (title, desc) in enumerate(steps):
        x = x0 + i * (w + gap)
        fill = PALETTE["purple"] if title == "HRG prior" else PALETTE["spine"] if title in {"Validate", "Retrieve"} else PALETTE["panel"]
        stroke = PALETTE["purple"] if title == "HRG prior" else PALETTE["spine"] if title in {"Validate", "Retrieve"} else PALETTE["line"]
        color = "white" if fill in {PALETTE["spine"], PALETTE["purple"]} else PALETTE["ink"]
        svg.rect(x, y, w, h, fill, stroke=stroke, sw=1.5, rx=8, shadow=True)
        svg.text(x + w / 2, y + 34, title, size=20, color=color, weight=700, anchor="middle")
        body_color = color if fill in {PALETTE["spine"], PALETTE["purple"]} else PALETTE["muted"]
        svg.multiline(x + 14, y + 62, wrap(desc, 22), size=13, color=body_color, line_h=16)
        centers.append((x + w / 2, y + h / 2))
        if i:
            px, py = centers[i - 1]
            svg.line(px + w / 2 - 4, py, x - 10, py, color=PALETTE["muted"], sw=2.5, arrow=True)

    svg.rect(120, 390, 1260, 155, PALETTE["panel2"], stroke=PALETTE["line"], sw=1.5, rx=8)
    svg.text(150, 425, "What becomes explainable?", size=22, weight=700, color=PALETTE["ink"])
    explain = [
        ("selected_entity", "topic node grounded in KG"),
        ("relation_labels", "ordered KB relation labels"),
        ("HRG signal", "grammar hit / ranking prior"),
        ("spine_edges", "printable evidence triples"),
        ("final_context", "small context sent to LLM"),
    ]
    for i, (a, b) in enumerate(explain):
        x = 160 + i * 238
        svg.pill(x, 456, a, PALETTE["spine"], w=180)
        svg.multiline(x, 508, wrap(b, 23), size=14, color=PALETTE["muted"], line_h=18)

    svg.rect(120, 595, 600, 145, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(150, 630, "Main evidence claim", size=21, weight=700, color=PALETTE["ok"])
    svg.multiline(150, 665, [
        "MetaQA legacy gpt-oss, 200 questions per hop",
        "BFS: 4415.83 context tokens",
        "HRG-Proposed-triple: 77.26 context tokens",
    ], size=17, color=PALETTE["ink"], line_h=24)

    svg.rect(780, 595, 600, 145, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(810, 630, "Claim boundary", size=21, weight=700, color=PALETTE["warn"])
    svg.multiline(810, 665, [
        "Main claim: compact, traceable evidence",
        "HRG supplies grammar and ranking signals",
        "Report model, sample size, and artifact source",
    ], size=17, color=PALETTE["ink"], line_h=24)
    svg.save(IMG_DIR / "01_method_pipeline.svg")


def fig_02_bfs_vs_spine() -> None:
    bfs = load_pickle(ART / "metaqa-vanilla-test/dumps/per_model/Baseline-BFS-gpt-oss@metaqa-vanilla-test/2-hop/q_0000.pkl")
    spine = load_pickle(ART / "metaqa-vanilla-test/dumps/per_model/HRG-Proposed-gpt-oss-json@metaqa-vanilla-test/2-hop/q_0000.pkl")
    svg = SVG(1500, 860, "Example retrieval: BFS neighborhood vs HRG-guided evidence")
    q = bfs["question"]
    svg.rect(50, 72, 1400, 72, PALETTE["panel"], stroke=PALETTE["line"], rx=8)
    svg.text(80, 116, f"Question: {q}", size=22, weight=700, color=PALETTE["ink"])

    svg.rect(60, 180, 660, 560, "#f8fafc", stroke=PALETTE["line"], rx=8)
    svg.text(90, 220, "Baseline-BFS context", size=23, weight=700, color=PALETTE["bfs"])
    svg.text(90, 248, "Collects all nearby facts under BFS depth", size=16, color=PALETTE["muted"])
    svg.pill(500, 205, f"{bfs['subgraph_size']} edges", PALETTE["bfs"], w=100)
    svg.pill(610, 205, f"{bfs['token_usage']['context_tokens']} ctx tokens", PALETTE["bfs"], w=125)

    pos = {
        "Something Borrowed": (360, 455),
        "John Krasinski": (145, 325),
        "Luke Greenfield": (595, 325),
        "Nobody Walks": (150, 470),
        "Leatherheads": (170, 610),
        "Away We Go": (370, 665),
        "2011": (605, 610),
    }
    bfs_edges = [
        ("Something Borrowed", "starred_actors", "John Krasinski"),
        ("Something Borrowed", "directed_by", "Luke Greenfield"),
        ("Nobody Walks", "starred_actors", "John Krasinski"),
        ("Leatherheads", "starred_actors", "John Krasinski"),
        ("Away We Go", "starred_actors", "John Krasinski"),
        ("Something Borrowed", "release_year", "2011"),
    ]
    for idx, (h, r, t) in enumerate(bfs_edges):
        x1, y1 = pos[h]
        x2, y2 = pos[t]
        important = t in {"John Krasinski", "Luke Greenfield"}
        if idx < 2:
            draw_edge(svg, x1, y1, x2, y2, r, PALETTE["spine"], sw=3)
        else:
            svg.line(x1, y1, x2, y2, color="#94a3b8", sw=1.7, arrow=True)
    for node, (x, y) in pos.items():
        fill = PALETTE["spine"] if node in {"John Krasinski", "Something Borrowed", "Luke Greenfield"} else "#94a3b8"
        draw_node(svg, x, y, node, fill, w=155 if node in {"Something Borrowed", "Luke Greenfield", "John Krasinski"} else 125)

    svg.rect(780, 180, 660, 560, "#f0fdfa", stroke="#99f6e4", rx=8)
    svg.text(810, 220, "HRG-Proposed context", size=23, weight=700, color=PALETTE["purple"])
    svg.text(810, 248, "Keeps compact evidence triples selected by KB + HRG signals", size=16, color=PALETTE["muted"])
    svg.pill(1200, 205, f"{spine['subgraph_size']} edges", PALETTE["spine"], w=100)
    svg.pill(1310, 205, f"{spine['token_usage']['context_tokens']} ctx tokens", PALETTE["spine"], w=125)

    draw_edge(svg, 1098, 390, 953, 390, "starred_actors", PALETTE["purple"], sw=4)
    draw_edge(svg, 1098, 505, 1243, 505, "directed_by", PALETTE["purple"], sw=4)
    draw_node(svg, 1180, 390, "Something Borrowed", PALETTE["spine2"], w=165)
    draw_node(svg, 875, 390, "John Krasinski", PALETTE["spine"], w=155)
    draw_node(svg, 1015, 505, "Something Borrowed", PALETTE["spine2"], w=165)
    draw_node(svg, 1320, 505, "Luke Greenfield", PALETTE["spine"], w=155)

    svg.rect(835, 560, 550, 120, "white", stroke="#99f6e4", rx=8)
    svg.text(860, 595, "Printable structure", size=19, weight=700, color=PALETTE["spine"])
    svg.text(860, 627, "selected_entity = John Krasinski", size=17, color=PALETTE["ink"])
    svg.text(860, 656, "relation labels = starred_actors, directed_by", size=17, color=PALETTE["ink"])
    svg.text(860, 685, "answers from directed_by evidence", size=17, color=PALETTE["ink"])

    svg.rect(280, 752, 940, 48, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(750, 783, "Same question, compact context, explicit KB triples", size=22, weight=700, color=PALETTE["ok"], anchor="middle")
    svg.save(IMG_DIR / "02_example_bfs_vs_spine_linda_evans.svg")


def fig_03_explainability_output() -> None:
    spine = load_pickle(ART / "metaqa-vanilla-test/dumps/per_model/HRG-Proposed-gpt-oss-json@metaqa-vanilla-test/2-hop/q_0000.pkl")
    svg = SVG(1500, 850, "What the explanation looks like for one question")
    svg.rect(60, 90, 1380, 90, PALETTE["panel"], stroke=PALETTE["line"], rx=8)
    svg.text(90, 128, "Question", size=18, color=PALETTE["muted"], weight=700)
    svg.text(90, 158, spine["question"], size=24, color=PALETTE["ink"], weight=700)

    cols = [
        (90, 250, 300, 125, "1. Topic entity", "John Krasinski"),
        (430, 250, 420, 125, "2. Relation labels", "starred_actors, directed_by"),
        (890, 250, 470, 125, "3. Answer source", "directed_by evidence"),
    ]
    for x, y, w, h, title, body in cols:
        svg.rect(x, y, w, h, "white", stroke=PALETTE["line"], rx=8, shadow=True)
        svg.text(x + 22, y + 38, title, size=18, color=PALETTE["muted"], weight=700)
        svg.multiline(x + 22, y + 78, wrap(body, 32), size=23, color=PALETTE["spine"], weight=700, line_h=28)

    svg.line(390, 312, 430, 312, color=PALETTE["muted"], sw=2.5, arrow=True)
    svg.line(850, 312, 890, 312, color=PALETTE["muted"], sw=2.5, arrow=True)

    svg.text(90, 455, "KB evidence triples printed by the system", size=23, weight=700, color=PALETTE["ink"])
    x, y = 90, 490
    svg.rect(x, y, 1280, 150, "#f0fdfa", stroke="#99f6e4", rx=8)
    headers = ["Head", "Relation", "Tail", "Role"]
    widths = [260, 300, 330, 280]
    cx = x + 30
    for i, h in enumerate(headers):
        svg.text(cx, y + 40, h, size=16, weight=700, color=PALETTE["muted"])
        cx += widths[i]
    rows = [
        ("Something Borrowed", "starred_actors", "John Krasinski", "matches topic entity"),
        ("Something Borrowed", "directed_by", "Luke Greenfield", "one answer evidence"),
    ]
    for r, row in enumerate(rows):
        yy = y + 78 + r * 40
        svg.line(x + 25, yy - 22, x + 1235, yy - 22, color="#ccfbf1", sw=1)
        cx = x + 30
        for i, cell in enumerate(row):
            svg.text(cx, yy, cell, size=18, color=PALETTE["ink"] if i != 1 else PALETTE["spine"], weight=700 if i == 1 else 400)
            cx += widths[i]

    svg.rect(90, 695, 1280, 78, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(120, 727, "Interpretability claim", size=20, weight=700, color=PALETTE["warn"])
    svg.text(120, 757, "This explains retrieved KB evidence and HRG signals, not hidden LLM reasoning.", size=20, color=PALETTE["ink"])
    svg.save(IMG_DIR / "03_explainability_output_linda_evans.svg")


def fig_04_compression_bars() -> None:
    res = get_metaqa_results()
    svg = SVG(1500, 850, "MetaQA legacy gpt-oss, 200-per-hop: context and edge compression")
    svg.text(80, 92, "Source: artifect_all/metaqa-vanilla-test; values match the gpt-oss column in the main result table.", size=18, color=PALETTE["muted"], weight=700)
    labels = ["BFS", "Spine Triple", "HRG Triple"]
    ctx = [res[l]["avg_ctx_tokens"] for l in labels]
    subg = [res[l]["avg_subgraph_size"] for l in labels]

    def bar_panel(x0: int, y0: int, title: str, values: List[float], suffix: str, colors: List[str]) -> None:
        svg.rect(x0, y0, 620, 565, PALETTE["panel"], stroke=PALETTE["line"], rx=8)
        svg.text(x0 + 30, y0 + 40, title, size=23, weight=700, color=PALETTE["ink"])
        maxv = max(values)
        base = y0 + 480
        chart_h = 350
        for i, (lab, val) in enumerate(zip(labels, values)):
            bx = x0 + 80 + i * 170
            bh = max(4, chart_h * (val / maxv))
            svg.rect(bx, base - bh, 95, bh, colors[i], rx=4)
            svg.text(bx + 47, base + 30, lab, size=15, color=PALETTE["ink"], weight=700, anchor="middle")
            svg.text(bx + 47, base - bh - 12, f"{val:.2f}{suffix}", size=16, color=colors[i], weight=700, anchor="middle")
        svg.line(x0 + 55, base, x0 + 560, base, color=PALETTE["line"], sw=2)

    bar_panel(70, 150, "Final context tokens", ctx, "", [PALETTE["bfs"], PALETTE["spine"], PALETTE["blue"]])
    bar_panel(810, 150, "Retrieved evidence edges", subg, "", [PALETTE["bfs"], PALETTE["spine"], PALETTE["blue"]])

    svg.rect(170, 730, 1160, 70, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(750, 773, "HRG-Proposed-triple keeps the evidence context at 1.75% of BFS for this gpt-oss run.", size=23, color=PALETTE["ok"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "04_metaqa_token_subgraph_compression.svg")


def fig_05_quality_vs_tokens() -> None:
    res = get_metaqa_results()
    svg = SVG(1500, 850, "MetaQA legacy gpt-oss, 200-per-hop: quality-cost trade-off")
    svg.rect(90, 120, 1120, 580, "white", stroke=PALETTE["line"], rx=8)
    svg.text(130, 160, "X = final context tokens, Y = answer-set F1; same source as the gpt-oss table column", size=18, color=PALETTE["muted"], weight=700)

    x_min, x_max = 0, 4600
    y_min, y_max = 0.54, 0.68
    plot = (170, 210, 950, 420)

    def sx(v: float) -> float:
        return plot[0] + (v - x_min) / (x_max - x_min) * plot[2]

    def sy(v: float) -> float:
        return plot[1] + plot[3] - (v - y_min) / (y_max - y_min) * plot[3]

    for t in [0, 1000, 2000, 3000, 4000]:
        x = sx(t)
        svg.line(x, plot[1], x, plot[1] + plot[3], color="#e2e8f0", sw=1)
        svg.text(x, plot[1] + plot[3] + 28, str(t), size=13, color=PALETTE["muted"], anchor="middle")
    for yv in [0.56, 0.58, 0.60, 0.62, 0.64, 0.66]:
        y = sy(yv)
        svg.line(plot[0], y, plot[0] + plot[2], y, color="#e2e8f0", sw=1)
        svg.text(plot[0] - 18, y + 5, f"{yv:.2f}", size=13, color=PALETTE["muted"], anchor="end")

    points = [
        ("BFS", res["BFS"]["avg_ctx_tokens"], res["BFS"]["answer_set_f1"], PALETTE["bfs"]),
        ("Spine JSON", res["Spine JSON"]["avg_ctx_tokens"], res["Spine JSON"]["answer_set_f1"], PALETTE["spine"]),
        ("HRG Triple", res["HRG Triple"]["avg_ctx_tokens"], res["HRG Triple"]["answer_set_f1"], PALETTE["purple"]),
    ]
    for name, xval, yval, color in points:
        x, y = sx(xval), sy(yval)
        svg.circle(x, y, 18, color, stroke="white", sw=3, shadow=True)
        dx = -120 if name == "BFS" else 25
        svg.rect(x + dx, y - 34, 160, 52, "white", stroke=color, rx=6)
        svg.text(x + dx + 12, y - 12, name, size=15, color=color, weight=700)
        svg.text(x + dx + 12, y + 9, f"F1 {yval:.4f}, {xval:.0f} tok", size=13, color=PALETTE["ink"])

    svg.line(plot[0], plot[1] + plot[3], plot[0] + plot[2], plot[1] + plot[3], color=PALETTE["ink"], sw=2)
    svg.line(plot[0], plot[1], plot[0], plot[1] + plot[3], color=PALETTE["ink"], sw=2)
    svg.text(plot[0] + plot[2] / 2, 690, "Average context tokens", size=16, color=PALETTE["ink"], weight=700, anchor="middle")
    svg.text(110, 415, "F1", size=16, color=PALETTE["ink"], weight=700, anchor="middle")

    svg.rect(1240, 160, 210, 440, "#f0fdfa", stroke="#99f6e4", rx=8)
    svg.text(1265, 200, "Takeaway", size=22, color=PALETTE["spine"], weight=700)
    svg.multiline(1265, 245, [
        "Main claim:",
        "compact",
        "traceable",
        "evidence.",
        "",
        "Report F1,",
        "coverage, and",
        "tokens together.",
    ], size=17, color=PALETTE["ink"], line_h=25)
    svg.save(IMG_DIR / "05_metaqa_quality_vs_tokens.svg")


def fig_06_dataset_matrix() -> None:
    svg = SVG(1500, 850, "Where HRG-guided executable evidence is most useful")
    rows = [
        ("MetaQA", "In scope", "Very strong", "Main compact-evidence evidence", PALETTE["ok"]),
        ("WikiMovies", "Near ceiling", "Small graph", "BFS already compact", PALETTE["gold"]),
        ("MLPQ", "Stress test", "HRG recovers evidence", "Cross-lingual relations", PALETTE["warn"]),
        ("KQAPro", "Stress test", "HRG recovers evidence", "Operators / qualifiers", PALETTE["red"]),
    ]
    x, y = 80, 150
    col_w = [210, 240, 280, 470]
    headers = ["Dataset", "Answer quality", "Token reduction", "How to present it"]
    svg.rect(x, y, sum(col_w), 64, PALETTE["ink"], rx=8)
    cx = x + 25
    for i, h in enumerate(headers):
        svg.text(cx, y + 40, h, size=18, color="white", weight=700)
        cx += col_w[i]
    yy = y + 82
    for ds, ans, tok, role, color in rows:
        svg.rect(x, yy - 10, sum(col_w), 92, "white", stroke=PALETTE["line"], rx=8, shadow=True)
        cx = x + 25
        svg.text(cx, yy + 38, ds, size=22, color=PALETTE["ink"], weight=700)
        cx += col_w[0]
        svg.pill(cx, yy + 14, ans, color, w=160)
        cx += col_w[1]
        svg.pill(cx, yy + 14, tok, color, w=205)
        cx += col_w[2]
        svg.multiline(cx, yy + 24, wrap(role, 42), size=18, color=PALETTE["ink"], line_h=24)
        yy += 110

    svg.rect(90, 675, 1290, 90, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(125, 710, "Recommended defense", size=22, color=PALETTE["warn"], weight=700)
    svg.text(125, 745, "Use MetaQA as the in-scope result; use MLPQ/KQAPro as diagnostic evidence-recovery stress tests.", size=20, color=PALETTE["ink"])
    svg.save(IMG_DIR / "06_dataset_takeaway_matrix.svg")


def fig_07_hop_analysis() -> None:
    res = get_metaqa_results()
    svg = SVG(1500, 850, "MetaQA hop analysis: compare BFS, Spine, and HRG-Proposed")
    svg.rect(90, 125, 1320, 590, "white", stroke=PALETTE["line"], rx=8)
    groups = ["1-hop", "2-hop", "3-hop"]
    methods = [("BFS", PALETTE["bfs"]), ("Spine JSON", PALETTE["spine"]), ("HRG Triple", PALETTE["purple"])]
    x0, base, chart_h = 180, 640, 430
    group_gap = 360
    bar_w = 58
    for gi, g in enumerate(groups):
        gx = x0 + gi * group_gap
        svg.text(gx + 92, base + 40, g, size=19, weight=700, color=PALETTE["ink"], anchor="middle")
        for mi, (m, color) in enumerate(methods):
            val = res[m]["results"][g]["em"]
            bh = chart_h * val
            bx = gx + mi * 70
            svg.rect(bx, base - bh, bar_w, bh, color, rx=4)
            svg.text(bx + bar_w / 2, base - bh - 10, f"{val:.3f}", size=14, color=color, weight=700, anchor="middle")
    for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = base - chart_h * yv
        svg.line(140, yy, 1250, yy, color="#e2e8f0", sw=1)
        svg.text(120, yy + 5, f"{yv:.2f}", size=13, color=PALETTE["muted"], anchor="end")
    svg.line(140, base, 1250, base, color=PALETTE["ink"], sw=2)
    svg.text(120, 385, "EM", size=16, color=PALETTE["ink"], weight=700, anchor="middle")

    lx, ly = 1030, 155
    for i, (m, color) in enumerate(methods):
        svg.rect(lx, ly + i * 34, 22, 22, color, rx=3)
        svg.text(lx + 34, ly + 18 + i * 34, m, size=16, color=PALETTE["ink"])
    svg.rect(230, 745, 1040, 62, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(750, 784, "3-hop shows why relation-chain evidence must be evaluated by hop depth.", size=23, color=PALETTE["ok"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "07_metaqa_hop_analysis.svg")


def fig_08_hrg_extraction() -> None:
    svg = SVG(1500, 900, "Offline HRG grammar extraction")
    svg.multiline(50, 78, wrap("This figure explains how structural grammar rules are learned from the KG before question answering.", 120), size=18, color=PALETTE["muted"])

    steps = [
        ("KG triples", "(head, relation, tail)\nMitchell, starred_actors, Linda Evans"),
        ("Labeled graph", "Triple graph\nedges keep relation labels"),
        ("Structural view", "Clique decomposition\nuses local topology"),
        ("Capped BFS samples", "Local subgraphs\navoid hub explosion"),
        ("Clique tree", "MCS ordering +\ntriangulation"),
        ("HRG rules", "bags -> production rules\ncount repeated structures"),
    ]
    x0, y0, w, h, gap = 55, 170, 205, 125, 30
    for i, (title, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        fill = PALETTE["spine"] if i == len(steps) - 1 else PALETTE["panel"]
        color = "white" if fill == PALETTE["spine"] else PALETTE["ink"]
        svg.rect(x, y0, w, h, fill, stroke=PALETTE["line"], rx=8, shadow=True)
        svg.text(x + w / 2, y0 + 34, title, size=19, weight=700, color=color, anchor="middle")
        svg.multiline(x + 16, y0 + 65, wrap(body, 24), size=13, color=color if fill == PALETTE["spine"] else PALETTE["muted"], line_h=17)
        if i:
            px = x0 + (i - 1) * (w + gap) + w
            svg.line(px + 6, y0 + h / 2, x - 8, y0 + h / 2, color=PALETTE["muted"], sw=2.5, arrow=True)

    svg.rect(100, 390, 570, 300, "#f8fafc", stroke=PALETTE["line"], rx=8)
    svg.text(130, 430, "How one bag becomes a rule", size=22, weight=700, color=PALETTE["ink"])
    draw_node(svg, 220, 560, "external node", PALETTE["blue"], w=150)
    draw_node(svg, 530, 505, "internal node", PALETTE["spine"], w=145)
    draw_node(svg, 530, 640, "child bag", PALETTE["purple"], w=120)
    draw_edge(svg, 295, 560, 457, 518, "terminal edge", PALETTE["spine"], sw=3)
    draw_edge(svg, 530, 527, 530, 620, "attachment", PALETTE["purple"], sw=2.5)
    svg.multiline(130, 660, [
        "external nodes = intersection with parent bag",
        "internal nodes = new nodes introduced in this bag",
        "terminal edges = KG edges assigned to this bag",
    ], size=15, color=PALETTE["muted"], line_h=21)

    svg.rect(760, 390, 640, 300, "#f0fdfa", stroke="#99f6e4", rx=8)
    svg.text(790, 430, "Resulting production rule", size=22, weight=700, color=PALETTE["spine"])
    rule_lines = [
        "LHS: N / rank",
        "",
        "RHS:",
        "  terminal_edges = [(a, relation, b), ...]",
        "  nonterminals = [N / rank with attachments]",
        "",
        "count = how often this structure appears",
    ]
    svg.multiline(800, 470, rule_lines, size=18, color=PALETTE["ink"], line_h=28)
    svg.rect(170, 765, 1160, 58, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(750, 802, "Use this as a structural prior, not as a full grammar decoder.", size=23, weight=700, color=PALETTE["warn"], anchor="middle")
    svg.save(IMG_DIR / "08_hrg_grammar_extraction.svg")


def fig_09_chain_validation() -> None:
    svg = SVG(1500, 850, "KB relation-label validation with actual triples")
    svg.rect(70, 90, 1360, 72, PALETTE["panel"], stroke=PALETTE["line"], rx=8)
    svg.text(100, 132, "Question: which person directed the movies starred by [John Krasinski]?", size=24, weight=700, color=PALETTE["ink"])

    svg.text(110, 225, "LLM candidate", size=22, weight=700, color=PALETTE["ink"])
    svg.rect(110, 255, 380, 150, "white", stroke=PALETTE["line"], rx=8, shadow=True)
    svg.text(135, 295, "entity = John Krasinski", size=19, color=PALETTE["spine"], weight=700)
    svg.text(135, 330, "relation labels =", size=18, color=PALETTE["muted"], weight=700)
    svg.text(135, 364, "starred_actors, directed_by", size=18, color=PALETTE["ink"])

    svg.line(510, 330, 610, 330, color=PALETTE["muted"], sw=3, arrow=True)
    svg.text(560, 305, "validate", size=16, color=PALETTE["muted"], anchor="middle")

    svg.rect(640, 225, 760, 245, "#f0fdfa", stroke="#99f6e4", rx=8, shadow=True)
    svg.text(675, 265, "Validated KB evidence triples", size=22, weight=700, color=PALETTE["spine"])
    draw_edge(svg, 1038, 330, 868, 330, "starred_actors", PALETTE["purple"], sw=4)
    draw_edge(svg, 983, 405, 1148, 405, "directed_by", PALETTE["purple"], sw=4)
    draw_node(svg, 1120, 330, "Something Borrowed", PALETTE["spine2"], w=165)
    draw_node(svg, 790, 330, "John Krasinski", PALETTE["spine"], w=155)
    draw_node(svg, 900, 405, "Something Borrowed", PALETTE["spine2"], w=165)
    draw_node(svg, 1225, 405, "Luke Greenfield", PALETTE["spine"], w=155)

    svg.rect(170, 500, 1160, 170, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(210, 540, "Validation rule", size=22, weight=700, color=PALETTE["ok"])
    svg.multiline(210, 580, [
        "The relation label must appear in actual KB triples.",
        "The topic entity and answer must be supported by retrieved evidence triples.",
        "HRG contributes grammar-hit and ranking signals; it does not create new relation names.",
    ], size=19, color=PALETTE["ink"], line_h=29)

    svg.rect(470, 720, 560, 54, PALETTE["spine"], rx=8)
    svg.text(750, 755, "valid evidence builds compact context", size=23, weight=700, color="white", anchor="middle")
    svg.save(IMG_DIR / "09_chain_validation_algorithm.svg")


def _load_report(tag: str) -> Dict[str, Any]:
    return load_json(ART / f"{tag}/results/benchmark_results.json")


def _find_method(data: Dict[str, Any], needle: str) -> Dict[str, Any]:
    for name, d in data.items():
        if needle in name and isinstance(d, dict):
            return d
    raise KeyError(needle)


def fig_10_failure_counts() -> None:
    rows = []
    for ds, tag in [
        ("MetaQA", "metaqa-vanilla-test"),
        ("WikiMovies", "wikimovies-wiki_entities-test"),
        ("MLPQ", "mlpq-en-zh-en-ills"),
        ("KQAPro", "kqapro-validation"),
    ]:
        d = _find_method(_load_report(tag), "HRG-Proposed-gpt-oss-json")
        f = d.get("failure_counts", {})
        rows.append((ds, f.get("ok", 0), f.get("no_candidates", 0), f.get("no_valid_chain", 0)))

    svg = SVG(1500, 850, "Diagnostic failures: when executable evidence needs stronger semantics")
    svg.rect(95, 125, 1280, 590, "white", stroke=PALETTE["line"], rx=8)
    headers = ["Dataset", "ok", "no_candidates", "no_valid_chain", "Interpretation"]
    widths = [180, 150, 210, 210, 470]
    x0, y0 = 130, 180
    cx = x0
    for i, h in enumerate(headers):
        svg.text(cx, y0, h, size=17, weight=700, color=PALETTE["muted"])
        cx += widths[i]

    interpretations = {
        "MetaQA": "High grammar-hit setting; compact evidence is usually available.",
        "WikiMovies": "Mostly stable because questions are simple 1-hop.",
        "MLPQ": "Cross-lingual grounding and relation alignment remain difficult.",
        "KQAPro": "Operators and qualifiers exceed simple relation-chain evidence.",
    }
    max_total = max(sum(r[1:]) for r in rows)
    yy = 235
    for ds, ok, nc, nv in rows:
        total = ok + nc + nv
        svg.line(120, yy - 34, 1330, yy - 34, color="#e2e8f0", sw=1)
        svg.text(x0, yy, ds, size=21, color=PALETTE["ink"], weight=700)
        vals = [(ok, PALETTE["ok"]), (nc, PALETTE["warn"]), (nv, PALETTE["red"])]
        cx = x0 + widths[0]
        for val, color in vals:
            svg.text(cx, yy, str(val), size=20, color=color, weight=700)
            cx += widths[1] if color == PALETTE["ok"] else widths[2]
        svg.multiline(x0 + sum(widths[:4]), yy - 18, wrap(interpretations[ds], 45), size=16, color=PALETTE["ink"], line_h=21)
        # stacked bar
        bx, by, bw, bh = 135, yy + 28, 900, 22
        start = bx
        for val, color in vals:
            ww = bw * (val / max(total, 1))
            svg.rect(start, by, ww, bh, color, rx=2)
            start += ww
        svg.text(bx + bw + 20, by + 17, f"n={total}", size=14, color=PALETTE["muted"])
        yy += 115

    svg.rect(190, 755, 1120, 58, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(750, 792, "Use this as diagnostic scope evidence: HRG recovers structure, while semantic filters remain future work.", size=20, color=PALETTE["warn"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "10_failure_counts_spine_correction.svg")


def fig_11_dataset_semantics_examples() -> None:
    svg = SVG(1500, 900, "Dataset semantics explain the results")
    examples = [
        ("MetaQA", "relation-chain friendly", "who directed films starred by [Linda Evans]?", "relation labels: starred_actors, directed_by", PALETTE["ok"]),
        ("WikiMovies", "mostly 1-hop", "what films did Michelle Trachtenberg star in?", "single KB relation label", PALETTE["gold"]),
        ("MLPQ", "cross-lingual path", "English question with en DBpedia / zh DBpedia evidence", "cross-lingual entity and relation alignment", PALETTE["warn"]),
        ("KQAPro", "semantic program", "How many counties have population > 7800 or < 40000000?", "operators: FindAll, FilterNum, Or, Count", PALETTE["red"]),
    ]
    x0, y0 = 70, 130
    for i, (ds, tag, q, structure, color) in enumerate(examples):
        x = x0 + (i % 2) * 710
        y = y0 + (i // 2) * 315
        svg.rect(x, y, 650, 260, "white", stroke=color, rx=8, shadow=True)
        svg.pill(x + 25, y + 24, ds, color, w=135)
        svg.text(x + 180, y + 45, tag, size=20, color=color, weight=700)
        svg.text(x + 30, y + 95, "Question style", size=16, color=PALETTE["muted"], weight=700)
        svg.multiline(x + 30, y + 126, wrap(q, 58), size=18, color=PALETTE["ink"], line_h=24)
        svg.text(x + 30, y + 190, "Required structure", size=16, color=PALETTE["muted"], weight=700)
        svg.multiline(x + 30, y + 222, wrap(structure, 58), size=18, color=color, weight=700, line_h=24)

    svg.rect(180, 790, 1140, 60, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(750, 828, "HRG-Proposed is strongest when question semantics align with reusable KG relation structures.", size=22, color=PALETTE["ok"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "11_dataset_semantics_examples.svg")


def fig_12_evaluation_design() -> None:
    svg = SVG(1500, 850, "Evaluation design: answer quality, evidence quality, and cost")
    svg.rect(80, 120, 310, 500, PALETTE["panel"], stroke=PALETTE["line"], rx=8, shadow=True)
    svg.text(115, 165, "Same question set", size=23, color=PALETTE["ink"], weight=700)
    svg.multiline(115, 210, ["MetaQA", "WikiMovies", "MLPQ", "KQAPro"], size=20, color=PALETTE["muted"], line_h=34)

    methods = [
        ("Baseline-BFS", PALETTE["bfs"]),
        ("Spine-Only", PALETTE["blue"]),
        ("Spine-Correction", PALETTE["spine"]),
        ("HRG-Proposed", PALETTE["purple"]),
    ]
    for i, (m, c) in enumerate(methods):
        y = 170 + i * 95
        svg.line(390, y, 545, y, color=PALETTE["muted"], sw=2.5, arrow=True)
        svg.rect(555, y - 36, 285, 72, "white", stroke=c, rx=8, shadow=True)
        svg.text(697, y + 8, m, size=20, color=c, weight=700, anchor="middle")

    svg.line(850, 315, 980, 315, color=PALETTE["muted"], sw=2.5, arrow=True)
    svg.rect(965, 120, 450, 510, "#f0fdfa", stroke="#99f6e4", rx=8, shadow=True)
    svg.text(1000, 165, "Measured outputs", size=23, color=PALETTE["spine"], weight=700)
    blocks = [
        ("Answer quality", "EM / Hits@1 / answer-set F1", PALETTE["blue"]),
        ("Evidence quality", "R@5 / faithfulness / answer-in-spine", PALETTE["spine"]),
        ("Cost", "context tokens / subgraph size / online proxy", PALETTE["purple"]),
        ("Failure modes", "no_candidates / no_valid_chain / OOM", PALETTE["warn"]),
    ]
    yy = 205
    for title, desc, color in blocks:
        svg.rect(1000, yy, 370, 78, "white", stroke=color, rx=8)
        svg.text(1022, yy + 30, title, size=18, color=color, weight=700)
        svg.multiline(1022, yy + 56, wrap(desc, 36), size=15, color=PALETTE["ink"], line_h=18)
        yy += 96

    svg.rect(230, 700, 1040, 62, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(750, 739, "Do not rank by EM alone: defend the quality-cost-evidence trade-off.", size=22, color=PALETTE["warn"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "12_evaluation_design.svg")


def fig_20_bfs_vs_hrg_process() -> None:
    svg = SVG(1500, 850, "Canonical BFS KG-RAG vs HRG-guided executable retrieval")
    svg.text(70, 92, "Baseline KG-RAG", size=24, color=PALETTE["bfs"], weight=700)
    svg.text(820, 92, "This work", size=24, color=PALETTE["spine"], weight=700)

    def box(x: int, y: int, title: str, body: str, color: str, fill: str = "white") -> None:
        svg.rect(x, y, 255, 90, fill, stroke=color, rx=8, shadow=True)
        svg.text(x + 18, y + 30, title, size=18, color=color, weight=700)
        svg.multiline(x + 18, y + 56, wrap(body, 28), size=14, color=PALETTE["ink"], line_h=17)

    # Baseline side
    left_steps = [
        ("Ground entity", "Linda Evans"),
        ("BFS expand", "all 1..h-hop neighbors"),
        ("Serialize", "hundreds of mixed triples"),
        ("LLM answer", "answer from noisy context"),
    ]
    for i, (title, body) in enumerate(left_steps):
        y = 135 + i * 130
        box(70, y, title, body, PALETTE["bfs"], "#f8fafc")
        if i:
            svg.line(198, y - 35, 198, y - 5, color=PALETTE["muted"], sw=2.5, arrow=True)

    # Visual BFS fanout
    cx, cy = 510, 260
    svg.circle(cx, cy, 28, PALETTE["bfs"])
    svg.text(cx, cy + 6, "E", size=18, color="white", weight=700, anchor="middle")
    for i, ang in enumerate([-70, -35, 0, 35, 70]):
        x1 = cx + 120 * math.cos(math.radians(ang))
        y1 = cy + 120 * math.sin(math.radians(ang))
        svg.line(cx + 28 * math.cos(math.radians(ang)), cy + 28 * math.sin(math.radians(ang)), x1, y1, color=PALETTE["bfs"], sw=2)
        svg.circle(x1, y1, 20, "#cbd5e1", stroke=PALETTE["bfs"])
        for j, off in enumerate([-18, 18]):
            x2 = x1 + 80
            y2 = y1 + off
            svg.line(x1 + 20, y1, x2 - 18, y2, color="#cbd5e1", sw=1.5)
            svg.circle(x2, y2, 15, "#e2e8f0", stroke="#94a3b8")
    svg.rect(420, 540, 285, 94, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(448, 575, "Failure mode", size=19, color=PALETTE["warn"], weight=700)
    svg.multiline(448, 604, ["Token explosion", "Answer may be present but buried", "No explicit verified path"], size=15, color=PALETTE["ink"], line_h=21)

    # HRG side
    right_steps = [
        ("Parse intent", "entity + relation-chain candidates"),
        ("KB validate", "execute each chain over triples"),
        ("HRG prior", "grammar hit + fallback/ranking signals"),
        ("Compact evidence", "only validated spine triples"),
    ]
    for i, (title, body) in enumerate(right_steps):
        y = 135 + i * 130
        color = PALETTE["purple"] if title == "HRG prior" else PALETTE["spine"]
        box(820, y, title, body, color, "#f0fdfa" if color == PALETTE["spine"] else "#f5f3ff")
        if i:
            svg.line(948, y - 35, 948, y - 5, color=PALETTE["muted"], sw=2.5, arrow=True)

    # HRG path visual
    draw_node(svg, 1200, 250, "Linda Evans", PALETTE["spine"], w=140)
    draw_node(svg, 1200, 370, "Mitchell", PALETTE["spine2"], w=130)
    draw_node(svg, 1200, 490, "Andrew McLaglen", PALETTE["spine"], w=170)
    draw_edge(svg, 1200, 278, 1200, 342, "starred_actors r-", PALETTE["purple"], sw=3)
    draw_edge(svg, 1200, 398, 1200, 462, "directed_by r+", PALETTE["purple"], sw=3)
    svg.rect(1100, 620, 300, 86, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(1125, 654, "What is added?", size=19, color=PALETTE["ok"], weight=700)
    svg.multiline(1125, 683, ["Per-hop validation", "Printable evidence path", "Grammar-aware diagnostics"], size=15, color=PALETTE["ink"], line_h=20)

    svg.rect(210, 750, 1080, 55, "#eef4f8", stroke=PALETTE["line"], rx=8)
    svg.text(750, 785, "BFS maximizes local coverage; HRG-guided retrieval constrains the context to executable, inspectable evidence.", size=20, color=PALETTE["ink"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "20_bfs_vs_hrg_process.svg")


def fig_21_mcs_triangulation_clique_tree() -> None:
    svg = SVG(1500, 850, "MCS, triangulation, and clique-tree grammar extraction")
    svg.text(60, 92, "Conceptual local graph example", size=20, color=PALETTE["muted"], weight=700)

    def small_node(x: int, y: int, label: str, color: str) -> None:
        svg.circle(x, y, 28, color, stroke="white", sw=2, shadow=True)
        svg.text(x, y + 6, label, size=16, color="white", weight=700, anchor="middle")

    # Original graph
    svg.rect(60, 130, 410, 280, "white", stroke=PALETTE["line"], rx=8, shadow=True)
    svg.text(90, 168, "1. Local skeleton", size=21, color=PALETTE["ink"], weight=700)
    coords = {"M": (180, 260), "D": (300, 200), "C": (390, 300), "G": (260, 350)}
    for a, b in [("M", "D"), ("D", "C"), ("C", "G"), ("G", "M")]:
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        svg.line(x1, y1, x2, y2, color=PALETTE["bfs"], sw=2.5)
    for n, (x, y) in coords.items():
        small_node(x, y, n, PALETTE["bfs"])
    svg.multiline(90, 430, ["M=movie, D=director,", "C=country, G=genre", "4-cycle has no chord"], size=15, color=PALETTE["muted"], line_h=20)

    # Triangulated graph
    svg.rect(545, 130, 410, 280, "white", stroke=PALETTE["line"], rx=8, shadow=True)
    svg.text(575, 168, "2. MCS + triangulation", size=21, color=PALETTE["ink"], weight=700)
    coords2 = {"M": (665, 260), "D": (785, 200), "C": (875, 300), "G": (745, 350)}
    for a, b in [("M", "D"), ("D", "C"), ("C", "G"), ("G", "M")]:
        x1, y1 = coords2[a]
        x2, y2 = coords2[b]
        svg.line(x1, y1, x2, y2, color=PALETTE["bfs"], sw=2.5)
    svg.line(coords2["D"][0], coords2["D"][1], coords2["G"][0], coords2["G"][1], color=PALETTE["warn"], sw=3, dash="6 5")
    for n, (x, y) in coords2.items():
        small_node(x, y, n, PALETTE["spine"])
    svg.text(795, 287, "fill edge", size=15, color=PALETTE["warn"], weight=700)
    svg.multiline(575, 430, ["The fill edge is only for", "decomposition; it is not a", "new KG triple."], size=15, color=PALETTE["muted"], line_h=20)

    # Clique tree
    svg.rect(1030, 130, 410, 280, "white", stroke=PALETTE["line"], rx=8, shadow=True)
    svg.text(1060, 168, "3. Clique tree", size=21, color=PALETTE["ink"], weight=700)
    svg.rect(1100, 230, 190, 68, "#f0fdfa", stroke=PALETTE["spine"], rx=8)
    svg.text(1195, 270, "{M, D, G}", size=21, color=PALETTE["spine"], weight=700, anchor="middle")
    svg.rect(1100, 330, 190, 68, "#f5f3ff", stroke=PALETTE["purple"], rx=8)
    svg.text(1195, 370, "{D, C, G}", size=21, color=PALETTE["purple"], weight=700, anchor="middle")
    svg.line(1195, 298, 1195, 330, color=PALETTE["muted"], sw=2.5, arrow=True)
    svg.text(1230, 320, "separator {D,G}", size=14, color=PALETTE["muted"])
    svg.multiline(1060, 430, ["Each bag becomes one local", "grammar rule with external", "attachment nodes."], size=15, color=PALETTE["muted"], line_h=20)

    svg.rect(95, 570, 1310, 135, "#ecfdf5", stroke="#86efac", rx=8)
    svg.text(125, 612, "Why this matters for retrieval", size=23, color=PALETTE["ok"], weight=700)
    svg.multiline(125, 650, [
        "Offline: repeated clique-tree bags become HRG-like structural priors.",
        "Online: matching relation-path signatures helps rank compact executable evidence instead of expanding every BFS neighbor.",
        "The triangulation edge is a decomposition artifact, not an added reverse relation or new KG fact.",
    ], size=18, color=PALETTE["ink"], line_h=27)
    svg.save(IMG_DIR / "21_mcs_triangulation_clique_tree.svg")


def fig_22_fallback_sources_examples() -> None:
    svg = SVG(1500, 850, "Fallback sources: failure forms and recovery methods")
    svg.text(60, 88, "Every fallback candidate is revalidated against the KG before retrieval.", size=20, color=PALETTE["muted"], weight=700)

    cols = [
        ("LLM correction", "Failure: relation alias or wrong order", "Recovery: ask LLM to repair failed hop, then KB-validate", PALETTE["blue"]),
        ("KG-valid fallback", "Failure: no parsed chain is executable", "Recovery: enumerate local executable chains from the grounded entity", PALETTE["spine"]),
        ("HRG-supported fallback", "Failure: many executable chains compete", "Recovery: use grammar hit, ordered path, and grammar score for ranking", PALETTE["purple"]),
        ("GrammarFirst diagnostic", "Failure: LLM hop/chain prior is wrong", "Recovery: HRG path-bank first, KG executable filtering second", PALETTE["warn"]),
    ]
    x0, y0, w, h = 55, 135, 335, 455
    for i, (title, fail, rec, color) in enumerate(cols):
        x = x0 + i * 360
        svg.rect(x, y0, w, h, "white", stroke=color, rx=8, shadow=True)
        svg.pill(x + 24, y0 + 25, title, color, w=210)
        svg.text(x + 24, y0 + 98, "Observed failure form", size=17, color=PALETTE["muted"], weight=700)
        svg.multiline(x + 24, y0 + 132, wrap(fail, 32), size=17, color=PALETTE["ink"], line_h=23)
        svg.text(x + 24, y0 + 220, "How it is recovered", size=17, color=PALETTE["muted"], weight=700)
        svg.multiline(x + 24, y0 + 254, wrap(rec, 32), size=17, color=PALETTE["ink"], line_h=23)
        svg.text(x + 24, y0 + 352, "Example", size=17, color=PALETTE["muted"], weight=700)
        if i == 0:
            ex = "directed -> directed_by"
        elif i == 1:
            ex = "Linda Evans -> starred_actors -> directed_by"
        elif i == 2:
            ex = "prefer common actor-movie-director pattern"
        else:
            ex = "entity seed -> HRG path-bank -> KG path"
        svg.multiline(x + 24, y0 + 386, wrap(ex, 32), size=16, color=color, weight=700, line_h=22)

    svg.rect(210, 675, 1080, 80, "#fff7ed", stroke="#fed7aa", rx=8)
    svg.text(750, 708, "Important boundary", size=21, color=PALETTE["warn"], weight=700, anchor="middle")
    svg.text(750, 739, "Executability solves syntax; semantic relevance still needs relation cues, answer type, and ranking.", size=19, color=PALETTE["ink"], weight=700, anchor="middle")
    svg.save(IMG_DIR / "22_fallback_sources_examples.svg")


def write_readme() -> None:
    text = """# Thesis Support Figures

Generated SVG figures for the HRG-guided KG-RAG thesis.

Recommended use:

1. `01_method_pipeline.svg` - method overview.
2. `02_example_bfs_vs_spine_linda_evans.svg` - show exactly how BFS vs HRG-Proposed retrieves evidence.
3. `03_explainability_output_linda_evans.svg` - show the printable explanation fields.
4. `04_metaqa_token_subgraph_compression.svg` - gpt-oss legacy 200-per-hop token and edge compression.
5. `05_metaqa_quality_vs_tokens.svg` - gpt-oss legacy 200-per-hop answer quality with fewer context tokens.
6. `06_dataset_takeaway_matrix.svg` - which datasets support the claim and which are limitations.
7. `07_metaqa_hop_analysis.svg` - why the 3-hop result matters.
8. `08_hrg_grammar_extraction.svg` - paper-style figure for offline grammar learning.
9. `09_chain_validation_algorithm.svg` - paper-style figure for executable chain validation.
10. `10_failure_counts_spine_correction.svg` - failure analysis across datasets.
11. `11_dataset_semantics_examples.svg` - why datasets differ.
12. `12_evaluation_design.svg` - evaluation is quality + efficiency + HRG-supported evidence.
20. `20_bfs_vs_hrg_process.svg` - canonical BFS KG-RAG compared with HRG-guided executable retrieval.
21. `21_mcs_triangulation_clique_tree.svg` - MCS, triangulation, clique tree, and HRG rule extraction intuition.
22. `22_fallback_sources_examples.svg` - failure forms and fallback recovery sources.

Notes:

- Figures 01-12 are thesis-support figures for method explanation, dataset diagnosis, and evaluation design.
- Numeric figures must state their source. Figures 04 and 05 use the legacy gpt-oss 200-per-hop MetaQA artifact and match the gpt-oss column in the method document.
"""
    (IMG_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    fig_01_pipeline()
    fig_02_bfs_vs_spine()
    fig_03_explainability_output()
    fig_04_compression_bars()
    fig_05_quality_vs_tokens()
    fig_06_dataset_matrix()
    fig_07_hop_analysis()
    fig_08_hrg_extraction()
    fig_09_chain_validation()
    fig_10_failure_counts()
    fig_11_dataset_semantics_examples()
    fig_12_evaluation_design()
    fig_20_bfs_vs_hrg_process()
    fig_21_mcs_triangulation_clique_tree()
    fig_22_fallback_sources_examples()
    write_readme()


if __name__ == "__main__":
    main()
