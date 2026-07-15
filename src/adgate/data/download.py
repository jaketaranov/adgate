"""Download real public datasets for the AdGate corpus.

Anyone who clones the repo runs:

    pip install -e ".[data]"
    python -m adgate.data.download --out data/

This is the single data source: one streaming pass over ESCI writes both
`ads.jsonl` (the ad inventory) and `queries.jsonl` (the labeled eval set),
with no intermediate files.

Source: Amazon ESCI "Shopping Queries Dataset" (Apache-2.0), via the
`tasksource/esci` Hugging Face mirror — ~2.6M human-graded query<->product
judgments (Exact/Substitute/Complement/Irrelevant). Products become the ad
inventory (brand/title/description map onto the Ad schema) and the ESCI
grades become graded relevance labels (E=2, S=1, C/I=0).

Streaming mode is used throughout: --max-ads/--max-judgments caps work
without downloading the full 1.2 GB dataset.

Note on categories: ESCI has no category column. `category` is left "" here;
deriving coarse categories (for the held-out-category trick) is future
work — either embedding clustering or the esci-s metadata extension
(github.com/shuttie/esci-s).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from ..schemas import Ad, IntentGrade, Message, QueryRecord

# ESCI grade -> AdGate graded relevance (2=strong, 1=acceptable, 0=irrelevant)
# tasksource/esci stores full-word labels ("Exact"), the official dump uses letters ("E");
# key on the first letter to accept both.
ESCI_GRADE = {"E": 2, "S": 1, "C": 0, "I": 0}


def esci_grade(label: str) -> int:
    return ESCI_GRADE.get((label or "")[:1].upper(), 0)

ESCI_DATASET = "tasksource/esci"

# Stratum 3 seed queries: clearly non-commercial. Deliberately spans
# definitions, code, math, trivia, emotional support, and translation.
# IMPROVEMENT: should come from a real dataset or LLM generation instead of
# a 30-item hardcoded list.
NONCOMMERCIAL_QUERIES = [
    "what is a derivative",
    "explain quantum entanglement simply",
    "how does photosynthesis work",
    "why is the sky blue",
    "write a python function to reverse a string",
    "help me debug this segfault in my c program",
    "summarize the plot of hamlet",
    "how do I apologize to my sister after an argument",
    "I feel anxious about my exam tomorrow",
    "what year did world war two end",
    "convert 100 fahrenheit to celsius",
    "what is the capital of mongolia",
    "prove that the square root of 2 is irrational",
    "difference between tcp and udp",
    "how many bones are in the human body",
    "translate hello how are you to japanese",
    "what rhymes with orange",
    "explain the offside rule in soccer",
    "who wrote the great gatsby",
    "how does a bill become a law",
    "what is the meaning of life",
    "how long do house cats usually live",
    "what is 17 times 23",
    "explain recursion like i am five",
    "give me a short history of the roman empire",
    "how do you say thank you in italian",
    "why do we dream",
    "grammar check this sentence: me and him went to store",
    "what does the mitochondria do",
    "explain how vaccines work",
]

SPLITS = (("calibration", 0.50), ("dev", 0.25), ("test", 0.25))


def product_to_ad(row: dict) -> Ad:
    """Map an ESCI product row onto the Ad schema."""
    desc = (row.get("product_description") or row.get("product_bullet_point") or "").strip()
    return Ad(
        ad_id=row["product_id"],
        brand=(row.get("product_brand") or "Unknown").strip(),
        title=(row.get("product_title") or "").strip(),
        body=desc[:300],
        cta="Shop now",
        click_url=f"https://www.amazon.com/dp/{row['product_id']}",
        category="",  # derived in Phase 0 (see module docstring)
        keywords=[k for k in [row.get("product_color")] if k],
    )


def download_esci(
    out_dir: Path,
    max_ads: int = 5000,
    max_judgments: int = 50000,
    locale: str = "us",
) -> tuple[int, int, dict[str, dict]]:
    """Stream ESCI (train split, reduced 'small_version' subset), writing
    unique products -> ads.jsonl. Judgments are grouped by query in memory
    (never written to disk) so build_esci_records() can turn them straight
    into queries.jsonl. Returns (n_ads, n_judgments, judgments_by_query).
    """
    from datasets import load_dataset  # deferred: requires the [data] extra

    ds = load_dataset(ESCI_DATASET, split="train", streaming=True)

    seen_products: set[str] = set()
    by_query: dict[str, dict] = defaultdict(lambda: {"query": "", "ads": {}})
    n_ads = n_judgments = 0
    with open(out_dir / "ads.jsonl", "w") as ads_f:
        for row in ds:
            if n_judgments >= max_judgments:
                break
            if row.get("product_locale") != locale or not row.get("small_version"):
                continue
            pid = row["product_id"]
            if pid not in seen_products:
                if n_ads >= max_ads:
                    continue  # ad cap hit: skip judgments for unseen products
                if not (row.get("product_title") or "").strip():
                    continue
                ads_f.write(product_to_ad(row).model_dump_json() + "\n")
                seen_products.add(pid)
                n_ads += 1
            q = by_query[str(row["query_id"])]
            q["query"] = row["query"]
            q["ads"][pid] = esci_grade(row["esci_label"])
            n_judgments += 1
    return n_ads, n_judgments, by_query


def build_esci_records(by_query: dict[str, dict]) -> list[QueryRecord]:
    """Group judgments by query -> one QueryRecord per unique query.
    should_show = any judged ad graded >= 1; queries where every judged ad
    is irrelevant become stratum 2 (commercially valid, nothing to show)."""
    records = []
    for qid, q in by_query.items():
        has_relevant = any(g >= 1 for g in q["ads"].values())
        records.append(QueryRecord(
            query_id=f"esci-{qid}",
            messages=[Message(role="user", content=q["query"])],
            stratum=1 if has_relevant else 2,
            # ESCI queries are product searches -> commercial research by
            # construction. IMPROVEMENT: grade each query with an LLM
            # instead of assuming; some are borderline TRANSACTIONAL.
            intent_grade=IntentGrade.RESEARCH,
            relevant_ad_ids=q["ads"],
            should_show=has_relevant,
        ))
    return records


def build_noncommercial_records() -> list[QueryRecord]:
    return [
        QueryRecord(
            query_id=f"nc-{i:03d}",
            messages=[Message(role="user", content=q)],
            stratum=3,
            intent_grade=IntentGrade.NONE,
            relevant_ad_ids={},
            should_show=False,
        )
        for i, q in enumerate(NONCOMMERCIAL_QUERIES)
    ]


def assign_splits(records: list[QueryRecord], seed: int = 13) -> list[QueryRecord]:
    """Stratified 50/25/25 split: shuffle within each stratum, then slice."""
    rng = random.Random(seed)
    by_stratum: dict[int, list[QueryRecord]] = defaultdict(list)
    for r in records:
        by_stratum[r.stratum].append(r)
    for stratum_records in by_stratum.values():
        stratum_records.sort(key=lambda r: r.query_id)  # deterministic before shuffle
        rng.shuffle(stratum_records)
        n = len(stratum_records)
        cut1 = int(n * SPLITS[0][1])
        cut2 = cut1 + int(n * SPLITS[1][1])
        for i, r in enumerate(stratum_records):
            r.split = "calibration" if i < cut1 else "dev" if i < cut2 else "test"
    return records


def main() -> None:
    parser = argparse.ArgumentParser(prog="adgate.data.download", description=__doc__)
    parser.add_argument("--out", default="data/", help="output directory")
    parser.add_argument("--max-ads", type=int, default=5000)
    parser.add_argument("--max-judgments", type=int, default=50000)
    parser.add_argument("--locale", default="us", help="ESCI product_locale filter")
    parser.add_argument("--seed", type=int, default=13, help="query split shuffle seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming ESCI ({ESCI_DATASET}) -> {out_dir}/ads.jsonl ...")
    n_ads, n_judgments, by_query = download_esci(
        out_dir, max_ads=args.max_ads, max_judgments=args.max_judgments, locale=args.locale
    )
    print(f"  {n_ads} ads, {n_judgments} graded judgments")

    records = build_esci_records(by_query)
    records += build_noncommercial_records()
    records = assign_splits(records, seed=args.seed)
    queries_out = out_dir / "queries.jsonl"
    with open(queries_out, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    by_stratum: dict[int, int] = defaultdict(int)
    for r in records:
        by_stratum[r.stratum] += 1
    print(f"  {len(records)} labeled queries -> {queries_out}")
    for s in sorted(by_stratum):
        print(f"    stratum {s}: {by_stratum[s]}")

    print("Done.")


if __name__ == "__main__":
    main()
