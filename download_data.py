#!/usr/bin/env python3
"""Standalone helper to populate the ``data/`` directory.

It first attempts to download the research-grade Enron-Spam corpus (Metsis
et al.). If the network is unavailable or a subset fails to download, it falls
back to the bundled synthetic sample emails so the service can always train.

Run it with::

    python download_data.py            # download all subsets + fallback seed
    python download_data.py --subsets enron1 enron2
    python download_data.py --seed-only
"""

from __future__ import annotations

import argparse
import sys

import data_loader


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare spam/ham training data.")
    parser.add_argument("--data-dir", default="data", help="Target data directory.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Enron subsets to download (default: enron1..enron6).",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Skip downloading and only write the bundled sample data.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-subset download timeout in seconds.",
    )
    args = parser.parse_args()

    if not args.seed_only:
        print("Downloading Enron-Spam dataset ...")
        results = data_loader.download_enron_dataset(
            args.data_dir, subsets=args.subsets, timeout=args.timeout
        )
        print(f"  succeeded: {results['succeeded']}")
        if results["failed"]:
            print(f"  failed: {results['failed']}")

    # Always ensure a baseline of data exists (no-op if real data is present).
    print("Ensuring baseline sample data exists ...")
    seeded = data_loader.seed_sample_data(args.data_dir)
    print(f"  ham files:  {seeded['ham']}")
    print(f"  spam files: {seeded['spam']}")

    emails, stats = data_loader.load_dataset(args.data_dir)
    print(f"Total emails available for training: {stats['total']} "
          f"(ham={stats['ham']}, spam={stats['spam']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
