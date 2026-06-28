from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "img" / "paper_figures"


PREAMBLE = r"""
\documentclass[tikz,border=12pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,fit,calc}
\definecolor{ink}{HTML}{172026}
\definecolor{muted}{HTML}{5F6B73}
\definecolor{line}{HTML}{CBD5E1}
\definecolor{panel}{HTML}{F8FAFC}
\definecolor{bfs}{HTML}{64748B}
\definecolor{spine}{HTML}{0F766E}
\definecolor{hrg}{HTML}{7C3AED}
\definecolor{warn}{HTML}{C2410C}
\definecolor{ok}{HTML}{15803D}
\definecolor{blue}{HTML}{2563EB}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\small, color=ink},
    label style={font=\small, color=ink},
    title style={font=\bfseries\small, color=ink},
    legend style={font=\footnotesize, draw=none, fill=none},
    grid=major,
    grid style={line!55},
    axis line style={line},
  }
}
\begin{document}
"""


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_compile(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / f"{name}.tex"
    tex_path.write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={OUT}",
            str(tex_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def fig13_architecture() -> None:
    body = r"""
\begin{tikzpicture}[
  font=\sffamily,
  box/.style={rounded corners=3pt, draw=line, thick, fill=panel, minimum width=3.05cm, minimum height=1.05cm, align=center, font=\small},
  hbox/.style={box, draw=hrg, fill=hrg!8},
  obox/.style={box, draw=spine, fill=spine!8},
  pbox/.style={rounded corners=3pt, draw=hrg, thick, fill=hrg!5, minimum width=4.9cm, minimum height=1.05cm, align=center, font=\small},
  arrow/.style={-{Latex[length=2.5mm]}, thick, draw=muted},
]
\node[font=\bfseries\Large, text=ink] at (7.0,4.95) {Offline HRG prior and online executable KGQA retrieval};
\node[font=\bfseries\small, text=hrg] at (0,4.10) {Offline prior extraction};
\node[font=\bfseries\small, text=spine] at (0,1.35) {Online QA retrieval};

\node[hbox] (kg) at (0,3.45) {KG local\\subgraphs};
\node[hbox] (mcs) at (3.45,3.45) {MCS +\\triangulation};
\node[hbox] (clique) at (6.90,3.45) {Clique-tree\\rules};
\node[hbox] (grammar) at (10.35,3.45) {HRG-like\\structural prior};
\draw[arrow] (kg) -- (mcs);
\draw[arrow] (mcs) -- (clique);
\draw[arrow] (clique) -- (grammar);

\node[pbox] (prioruse) at (10.35,2.05) {Use prior for candidate\\selection, fallback, and ranking};
\draw[arrow, draw=hrg] (grammar.south) -- (prioruse.north);

\node[obox] (q) at (0,0.70) {Question};
\node[obox] (parse) at (3.45,0.70) {Entity +\\relation chain};
\node[obox] (valid) at (6.90,0.70) {KB validation\\strict spine};
\node[obox] (evid) at (10.35,0.70) {Retrieved\\KG evidence};
\node[obox] (ans) at (13.80,0.70) {LLM answer};
\draw[arrow] (q) -- (parse);
\draw[arrow] (parse) -- (valid);
\draw[arrow] (valid) -- (evid);
\draw[arrow] (evid) -- (ans);
\draw[arrow, draw=hrg] (prioruse.south) -- (evid.north);

\node[draw=warn, fill=warn!8, rounded corners=3pt, align=center, text=ink, minimum width=14.6cm, minimum height=0.72cm, font=\small] at (6.90,-0.85)
{Claim boundary: HRG is a soft structural prior for compact evidence retrieval, not a hard decoder or a universal accuracy guarantee.};
\end{tikzpicture}
"""
    write_compile("13_offline_online_architecture", body)


def fig14_grammar_stats() -> None:
    datasets = ["MetaQA", "WikiMovies", "MLPQ", "KQAPro"]
    rules = [304, 335, 398, 187]
    patterns = [247, 248, 353, 185]
    rels = [9, 10, 262, 300]
    coords_rules = " ".join(f"({d},{v})" for d, v in zip(datasets, rules))
    coords_patterns = " ".join(f"({d},{v})" for d, v in zip(datasets, patterns))
    coords_rels = " ".join(f"({d},{v})" for d, v in zip(datasets, rels))
    body = rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  ybar,
  bar width=10pt,
  width=15.5cm,
  height=7.8cm,
  ymin=0,
  ymax=460,
  enlarge x limits=0.18,
  ylabel={{Count}},
  symbolic x coords={{MetaQA,WikiMovies,MLPQ,KQAPro}},
  xtick=data,
  x tick label style={{rotate=0, font=\small}},
  legend columns=3,
  legend style={{at={{(0.5,-0.14)}}, anchor=north, font=\footnotesize}},
  title={{Offline grammar extraction statistics}},
]
\addplot+[fill=hrg!72, draw=hrg] coordinates {{{coords_rules}}};
\addplot+[fill=blue!65, draw=blue] coordinates {{{coords_patterns}}};
\addplot+[fill=spine!65, draw=spine] coordinates {{{coords_rels}}};
\legend{{Extracted rules, Unique relation patterns, Unique relations}}
\end{{axis}}
\end{{tikzpicture}}
"""
    write_compile("14_offline_grammar_statistics", body)


def fig15_perturbation() -> None:
    # Rule count trends and relation retention trends.
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  width=15.5cm,
  height=6.7cm,
  xmin=0, xmax=30,
  ymin=0, ymax=580,
  xlabel={Node drop ratio (\%)},
  ylabel={Extracted rule count},
  title={Offline HRG rule count under node deletion},
  legend columns=4,
  legend style={at={(0.5,-0.18)}, anchor=north, font=\footnotesize},
]
\addplot+[mark=*, thick, color=hrg] coordinates {(0,304) (10,326) (20,368) (30,433)};
\addplot+[mark=square*, thick, color=spine] coordinates {(0,335) (10,314) (20,360) (30,376)};
\addplot+[mark=triangle*, thick, color=blue] coordinates {(0,398) (10,467) (20,453) (30,526)};
\addplot+[mark=diamond*, thick, color=warn] coordinates {(0,187) (10,343) (20,350) (30,288)};
\legend{MetaQA,WikiMovies,MLPQ,KQAPro}
\end{axis}
\begin{axis}[
  xshift=0cm,
  yshift=-8.1cm,
  width=15.5cm,
  height=6.7cm,
  xmin=0, xmax=30,
  ymin=0, ymax=1.12,
  xlabel={Node drop ratio (\%)},
  ylabel={Retained clean relations},
  title={Relation vocabulary retention under node deletion},
  legend columns=4,
  legend style={at={(0.5,-0.18)}, anchor=north, font=\footnotesize},
]
\addplot+[mark=*, thick, color=hrg] coordinates {(0,1.000) (10,1.000) (20,1.000) (30,0.889)};
\addplot+[mark=square*, thick, color=spine] coordinates {(0,1.000) (10,0.900) (20,1.000) (30,1.000)};
\addplot+[mark=triangle*, thick, color=blue] coordinates {(0,1.000) (10,0.443) (20,0.458) (30,0.523)};
\addplot+[mark=diamond*, thick, color=warn] coordinates {(0,1.000) (10,0.597) (20,0.550) (30,0.487)};
\legend{MetaQA,WikiMovies,MLPQ,KQAPro}
\end{axis}
\end{tikzpicture}
"""
    write_compile("15_grammar_perturbation_trends", body)


def fig16_evidence_coverage() -> None:
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  ybar,
  bar width=9pt,
  width=15.8cm,
  height=7.6cm,
  ymin=0,
  ymax=0.58,
  enlarge x limits=0.16,
  ylabel={Coverage score},
  symbolic x coords={MLPQ RetF1,MLPQ AnsSpine,KQA RetF1,KQA AnsSpine},
  xtick=data,
  x tick label style={font=\footnotesize, align=center},
  legend columns=2,
  legend style={at={(0.5,-0.14)}, anchor=north, font=\footnotesize},
  title={HRG-Proposed improves evidence coverage in hard cases},
]
\addplot+[fill=spine!65, draw=spine] coordinates {(MLPQ RetF1,0.141) (MLPQ AnsSpine,0.225) (KQA RetF1,0.018) (KQA AnsSpine,0.026)};
\addplot+[fill=hrg!70, draw=hrg] coordinates {(MLPQ RetF1,0.167) (MLPQ AnsSpine,0.283) (KQA RetF1,0.105) (KQA AnsSpine,0.152)};
\legend{Spine-Correction-triple, HRG-Proposed-triple}
\end{axis}
\end{tikzpicture}
"""
    write_compile("16_evidence_coverage_hard_cases", body)


def fig17_token_proxy() -> None:
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  ybar,
  bar width=10pt,
  width=15.8cm,
  height=8.0cm,
  ymin=0,
  ymax=280,
  ytick={0,50,100,150,200,250},
  scaled y ticks=false,
  enlarge x limits=0.15,
  ylabel={Total online token proxy vs BFS (\%)},
  symbolic x coords={MetaQA,WikiMovies,MLPQ,KQAPro},
  xtick=data,
  ymajorgrids=true,
  legend columns=2,
  legend style={at={(0.5,-0.14)}, anchor=north, font=\footnotesize},
  title={End-to-end online token proxy ratio, not only final context},
]
\addplot+[fill=bfs!65, draw=bfs] coordinates {(MetaQA,100.0) (WikiMovies,100.0) (MLPQ,100.0) (KQAPro,100.0)};
\addplot+[fill=spine!65, draw=spine] coordinates {(MetaQA,15.7) (WikiMovies,196.3) (MLPQ,15.4) (KQAPro,29.5)};
\addplot+[fill=hrg!70, draw=hrg] coordinates {(MetaQA,17.0) (WikiMovies,252.1) (MLPQ,23.4) (KQAPro,65.5)};
\legend{Baseline-BFS, Spine-Correction-triple, HRG-Proposed-triple}
\end{axis}
\end{tikzpicture}
"""
    write_compile("17_online_token_proxy", body)


def fig18_bootstrap() -> None:
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  width=15.8cm,
  height=8.9cm,
  xmin=-0.06, xmax=0.155,
  ymin=0, ymax=7,
  xlabel={HRG-Proposed minus Spine-Correction},
  xtick={-0.05,0,0.05,0.10,0.15},
  xticklabels={-0.05,0,0.05,0.10,0.15},
  ytick={1,2,3,4,5,6},
  yticklabels={MetaQA EM,MetaQA ans-spine,MLPQ EM,MLPQ ans-spine,KQAPro EM,KQAPro ans-spine},
  yticklabel style={font=\footnotesize, align=right, text width=2.7cm},
  title={Paired bootstrap 95\% CI for HRG hard-case effects},
  xmajorgrids=true,
  ymajorgrids=false,
]
\addplot[draw=black, dashed] coordinates {(0,0) (0,7)};
\addplot+[only marks, mark=*, color=hrg, error bars/.cd, x dir=both, x explicit]
coordinates {
  (-0.0013,1) +- (0.0083,0)
  (-0.0370,2) +- (0.0142,0)
  (0.0250,3) +- (0.0113,0)
  (0.0576,4) +- (0.0140,0)
  (0.0530,5) +- (0.0100,0)
  (0.1256,6) +- (0.0138,0)
};
\end{axis}
\end{tikzpicture}
"""
    write_compile("18_bootstrap_hard_case_effects", body)


def fig19_prior_pilot() -> None:
    body = r"""
\begin{tikzpicture}
\begin{axis}[
  ybar,
  bar width=8pt,
  width=15.8cm,
  height=8.0cm,
  ymin=0,
  ymax=0.86,
  enlarge x limits=0.15,
  ylabel={Candidate retrieval F1},
  symbolic x coords={MetaQA,WikiMovies,MLPQ,KQAPro},
  xtick=data,
  legend columns=3,
  legend style={at={(0.5,-0.14)}, anchor=north, font=\footnotesize},
  title={Candidate-level prior pilot: HRG score vs relation unigram},
]
\addplot+[fill=hrg!70, draw=hrg] coordinates {(MetaQA,0.5140) (WikiMovies,0.7558) (MLPQ,0.1582) (KQAPro,0.1375)};
\addplot+[fill=blue!60, draw=blue] coordinates {(MetaQA,0.4900) (WikiMovies,0.7289) (MLPQ,0.1494) (KQAPro,0.1195)};
\addplot+[fill=ok!60, draw=ok] coordinates {(MetaQA,0.5309) (WikiMovies,0.7558) (MLPQ,0.1860) (KQAPro,0.1538)};
\legend{HRG score only, Relation unigram, LLM confidence}
\end{axis}
\end{tikzpicture}
"""
    write_compile("19_prior_pilot", body)


def write_index() -> None:
    lines = [
        "# Paper Figures",
        "",
        "| File | Use in thesis | Purpose |",
        "|---|---|---|",
        "| `13_offline_online_architecture.pdf` | Sections 1.1.2, Method overview | Shows the two-part thesis structure: offline HRG prior and online KGQA retrieval. |",
        "| `14_offline_grammar_statistics.pdf` | Section 21.A | Shows extracted rule counts, unique patterns, and relation coverage. |",
        "| `15_grammar_perturbation_trends.pdf` | Section 21.B | Shows grammar robustness under 10/20/30% node deletion. |",
        "| `16_evidence_coverage_hard_cases.pdf` | Section 21.6 | Shows HRG-Proposed improves retrieval F1 and answer-in-spine on MLPQ/KQAPro. |",
        "| `17_online_token_proxy.pdf` | Section 21.6.1 | Shows end-to-end online token proxy ratio, not just final context. |",
        "| `18_bootstrap_hard_case_effects.pdf` | Section 21.6.2 | Shows paired bootstrap confidence intervals for HRG vs Spine-Correction. |",
        "| `19_prior_pilot.pdf` | Section 21.6.3 | Shows candidate-level HRG score vs relation unigram prior pilot. |",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    fig13_architecture()
    fig14_grammar_stats()
    fig15_perturbation()
    fig16_evidence_coverage()
    fig17_token_proxy()
    fig18_bootstrap()
    fig19_prior_pilot()
    write_index()


if __name__ == "__main__":
    main()
