#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations

from drylab_common import best_by_method_target, best_of_k, read_csv, ranking_score, write_csv


def head_to_head(rows):
    best = best_by_method_target(rows)
    targets = sorted({target for target, _method in best})
    methods = sorted({method for _target, method in best})
    out = []
    for method_a, method_b in combinations(methods, 2):
        compared = 0
        wins_a = 0
        for target in targets:
            a = best.get((target, method_a))
            b = best.get((target, method_b))
            if not a or not b:
                continue
            compared += 1
            wins_a += int(ranking_score(a) > ranking_score(b))
        if compared:
            out.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "targets_compared": compared,
                    "wins_a": wins_a,
                    "wins_b": compared - wins_a,
                    "win_rate_a": wins_a / compared,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NACraft dry-lab ranking, best-of-K and head-to-head summaries.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--head-to-head-output", required=True)
    parser.add_argument("--best-of-k-output", required=True)
    args = parser.parse_args()

    rows = read_csv(args.metrics)
    write_csv(args.head_to_head_output, head_to_head(rows))
    write_csv(args.best_of_k_output, best_of_k(rows))


if __name__ == "__main__":
    main()
