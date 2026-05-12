from __future__ import annotations

import argparse
from pathlib import Path

from LLM_inference_benchmark.dataset_utils import iter_kb_triples


SAFE_MULTI_VALUE_RELATIONS = {
    "directed_by",
    "written_by",
    "starred_actors",
    "has_tags",
    "has_genre",
    "in_language",
}


def normalize_wikimovies_kb(
    input_path: str,
    output_path: str,
    split_relations: set[str],
) -> dict[str, int]:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    input_triples = 0
    output_triples = 0
    expanded_triples = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for h, r, t in iter_kb_triples(str(src)):
            input_triples += 1

            if r in split_relations and "," in t:
                parts = [part.strip() for part in t.split(",") if part.strip()]
                if len(parts) > 1:
                    for part in parts:
                        fout.write(f"{h}|{r}|{part}\n")
                        output_triples += 1
                    expanded_triples += 1
                    continue

            fout.write(f"{h}|{r}|{t}\n")
            output_triples += 1

    return {
        "input_triples": input_triples,
        "output_triples": output_triples,
        "expanded_source_triples": expanded_triples,
        "net_new_triples": output_triples - input_triples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize WikiMovies KB by splitting safe multi-value tails into atomic triples."
    )
    parser.add_argument(
        "--input",
        default="Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt",
        help="Path to the original WikiMovies KB file.",
    )
    parser.add_argument(
        "--output",
        default="Datasets/WikiMovies/movieqa/knowledge_source/wiki_entities/wiki_entities_kb_normalized.txt",
        help="Output path for the normalized KB.",
    )
    parser.add_argument(
        "--relations",
        nargs="*",
        default=sorted(SAFE_MULTI_VALUE_RELATIONS),
        help="Relations whose comma-separated tails should be split into atomic triples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = normalize_wikimovies_kb(
        input_path=args.input,
        output_path=args.output,
        split_relations=set(args.relations),
    )

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"split_relations={sorted(set(args.relations))}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
