#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drylab_common import rcsb_entries_to_manifest_candidates, write_csv


GRAPHQL_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    rcsb_accession_info{initial_release_date}
    polymer_entities{
      entity_poly{
        rcsb_entity_polymer_type
        pdbx_seq_one_letter_code_can
      }
      rcsb_polymer_entity_container_identifiers{
        auth_asym_ids
        asym_ids
      }
    }
  }
}
"""


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RCSB polymer metadata and build manifest candidate rows.")
    parser.add_argument("--hits-json", required=True)
    parser.add_argument("--polymer-type", choices=["rna", "dna"], required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    try:
        import requests
    except ImportError as exc:
        raise SystemExit("requests is required for RCSB metadata fetching") from exc
    hits = json.loads(Path(args.hits_json).read_text())
    ids = [item["identifier"] for item in hits.get("result_set", [])]
    if args.limit:
        ids = ids[: args.limit]
    rows = []
    for batch in chunked(ids, args.batch_size):
        response = requests.post(
            "https://data.rcsb.org/graphql",
            json={"query": GRAPHQL_QUERY, "variables": {"ids": batch}},
            timeout=60,
        )
        response.raise_for_status()
        rows.extend(rcsb_entries_to_manifest_candidates(response.json(), args.polymer_type, args.structure_root))
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
