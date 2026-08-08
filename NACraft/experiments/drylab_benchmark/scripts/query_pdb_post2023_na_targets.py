#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_query(polymer_type: str, release_after: str) -> dict:
    na_entity = "RNA" if polymer_type == "rna" else "DNA"
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater",
                        "value": release_after,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": na_entity,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query RCSB for post-2023 protein-RNA/DNA complexes.")
    parser.add_argument("--polymer-type", choices=["rna", "dna"], required=True)
    parser.add_argument("--release-after", default="2023-01-13")
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-query-only", action="store_true")
    args = parser.parse_args()

    query = build_query(args.polymer_type, args.release_after)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_query_only:
        output.write_text(json.dumps(query, indent=2))
        return
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("requests is unavailable; rerun with --write-query-only or install requests") from exc
    response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=60)
    response.raise_for_status()
    output.write_text(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
