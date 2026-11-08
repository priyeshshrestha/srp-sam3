"""Draw the fixed episode list that every model will run.

    python scripts/make_episodes.py --root ./data --band_order rgb --out outputs/episodes.json

--band_order has no default on purpose: decide it with scripts/check_bands.py first.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.flair import FLAIR_CLASSES  # noqa: E402
from datasets.flair_rgb import BAND_ORDERS, build_class_counts, discover_patches, domains_of  # noqa: E402
from evaluation.episodes import (  # noqa: E402
    DOMAIN_MODES, episodes_hash, load_episodes, make_episodes, save_episodes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--band_order", required=True, choices=BAND_ORDERS)
    ap.add_argument("--out", default="outputs/episodes.json")
    ap.add_argument("--classes", type=int, nargs="*", default=list(range(1, 13)),
                    help="class IDs to evaluate (default 1-12, the FLAIR baseline classes)")
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--domain_mode", choices=DOMAIN_MODES + ("auto",), default="auto",
                    help="cross = support and query from different FLAIR domains; same = same domain; "
                         "auto = cross if the data has >= 2 domains, otherwise same (with a warning)")
    ap.add_argument("--min_frac", type=float, default=0.02)
    ap.add_argument("--max_frac", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_shot_max", type=int, default=5,
                    help="supports stored per episode; K-shot runs use the first K")
    ap.add_argument("--counts_cache", default="outputs/class_counts.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different episodes file")
    args = ap.parse_args()

    rels = discover_patches(args.root)
    doms = domains_of(rels)
    n_domains = len(set(doms.values()))
    print(f"{len(rels)} patches, {n_domains} domain(s)")

    mode = args.domain_mode
    if mode == "auto":
        mode = "cross" if n_domains >= 2 else "same"
        if mode == "same":
            print("\n  *** WARNING: only ONE domain in this data, so episodes are SAME-domain. Support and\n"
                  "  *** query come from the same region/season; numbers will be optimistic and cannot\n"
                  "  *** speak to geographic transfer. Add more FLAIR domains for a cross-domain result.\n")

    # Identity of this configuration. An existing file is reused only if it matches exactly,
    # so a stale file (old band order, other data, other seed) can never be used by accident.
    identity = {"band_order": args.band_order, "classes": args.classes, "n_per_class": args.n_per_class,
                "domain_mode": mode, "min_frac": args.min_frac, "max_frac": args.max_frac,
                "seed": args.seed, "k_shot_max": args.k_shot_max,
                "n_patches": len(rels), "n_domains": n_domains}
    if os.path.exists(args.out):
        old = load_episodes(args.out)["meta"]
        differs = {k: (old.get(k), v) for k, v in identity.items() if old.get(k) != v}
        if not differs:
            print(f"Reusing existing {args.out} (configuration identical).")
            return
        if not args.force:
            sys.exit(f"ERROR: {args.out} exists with a DIFFERENT configuration: {differs}\n"
                     f"Delete it (and outputs/runs/*) or pass --force to regenerate. Results computed on the "
                     f"old episodes must not be mixed with new ones.")
        print(f"--force: regenerating {args.out}; differences: {differs}")

    counts = build_class_counts(args.root, rels, cache=args.counts_cache)
    payload = make_episodes(counts, doms, args.classes, args.n_per_class, args.seed,
                            mode, args.min_frac, args.max_frac, k_shot_max=args.k_shot_max)
    if not payload["episodes"]:
        sys.exit(f"ERROR: no episodes could be drawn (domain_mode={mode}). Skipped: {payload['skipped']}")
    meta = {**identity, "requested_domain_mode": args.domain_mode,
            "episodes_sha": episodes_hash(payload["episodes"]), "skipped": payload["skipped"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_episodes(args.out, payload, meta)

    per = Counter(e["class_id"] for e in payload["episodes"])
    print(f"\nWrote {len(payload['episodes'])} episodes to {args.out}")
    for c in args.classes:
        note = f"   SKIPPED/SHORT: {payload['skipped'][c]}" if c in payload["skipped"] else ""
        print(f"  class {c:2d} {FLAIR_CLASSES.get(c, '?'):<24}{per.get(c, 0):>4}{note}")


if __name__ == "__main__":
    main()
