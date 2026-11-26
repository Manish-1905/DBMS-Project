#!/usr/bin/env python3
"""
Controlled schema distractors. (Roadmap Part 4.5)

Pads a real schema with fake-but-plausible node labels and relationship types,
so the LLM sees a large, noisy schema while the question and the graph stay
the same:

    share = fraction of the final schema elements (labels + relationship types)
            that are distractors.  ER has 4 labels + 3 relationship types = 7:

        share 0.40 ->   5 distractors  (12 elements)
        share 0.60 ->  11 distractors  (18 elements)
        share 0.80 ->  28 distractors  (35 elements)
        share 0.95 -> 133 distractors (140 elements)

Only the schema TEXT given to the LLM changes. The Neo4j graph is untouched,
so gold queries still run; an LLM query that uses a distractor simply returns
nothing (or fails) and is scored as wrong.

Distractors are:
  * "hard": names that share words with real labels (UserProfile, MovieReview,
    GenreTag, IpBlock ...) and common property names (name, title, year ...);
  * "easy": unrelated streaming/e-commerce/ops concepts (Invoice, Warehouse ...).
Distractor relationships always touch at least one distractor label, so the
real labels are never connected to each other in a new way.

Deterministic: the same (schema, share, seed) always gives the same schema.

    uv run distractors.py --share 0.8                      # print a summary
    uv run distractors.py --share 0.8 --out noisy_er.json  # write the schema
"""

import argparse
import json
import random

LEVELS = (0.0, 0.4, 0.6, 0.8, 0.95)

# Words joined to real label names to make "hard" distractors (UserProfile ...).
HARD_SUFFIXES = ["Profile", "Review", "Tag", "Group", "Session", "Rating", "Block",
                 "Account", "List", "Award", "Region", "Archive", "Clip", "Note"]

EASY_LABELS = [
    "Actor", "Director", "Studio", "Award", "Trailer", "Episode", "Season", "Playlist",
    "Subscription", "Plan", "Invoice", "Payment", "Device", "Country", "City",
    "Language", "Subtitle", "Review", "Rating", "Comment", "Tag", "Keyword",
    "Franchise", "Character", "Soundtrack", "Song", "Artist", "Album", "Festival",
    "Critic", "Channel", "Broadcast", "Advertiser", "Campaign", "Coupon", "Order",
    "Product", "Warehouse", "Supplier", "Shipment", "Ticket", "Cinema", "Screening",
    "Seat", "Employee", "Department", "Project", "Server", "Datacenter", "Alert",
    "Incident", "Log", "Certificate", "Domain", "Router", "Firewall", "Vendor",
    "License", "Contract", "Survey", "Question", "Answer", "Badge", "Level", "Quest",
    "Team", "League", "Match", "Venue", "Sponsor", "Podcast", "Host", "Guest",
]

REL_VERBS = [
    "RATED", "REVIEWED", "FOLLOWS", "DIRECTED", "ACTED_IN", "PRODUCED", "BELONGS_TO",
    "TAGGED", "RECOMMENDS", "PURCHASED", "SUBSCRIBED_TO", "STREAMED", "LIKES",
    "SHARED", "CREATED", "OWNS", "LOCATED_IN", "SPEAKS", "HAS_SUBTITLE", "NOMINATED_FOR",
    "WON", "FEATURES", "PART_OF", "SCREENED_AT", "SOLD_BY", "SHIPPED_TO", "PAID_WITH",
    "LOGGED_FROM", "CONNECTED_TO", "REPORTED", "ASSIGNED_TO", "MANAGES", "WORKS_FOR",
    "SPONSORS", "HOSTS", "APPEARS_IN", "COMMENTED_ON", "ANSWERED", "EARNED", "PLAYED",
    "VISITED", "BOOKED", "CANCELLED", "UPGRADED_TO", "DOWNLOADED", "BOOKMARKED",
    "TRANSLATED", "LICENSED_BY", "DISTRIBUTED_BY", "FLAGGED", "BLOCKED", "INVITED",
    "MENTIONED", "QUOTED", "REMIXED", "SAMPLED", "COVERED", "AIRED_ON", "PREVIEWED",
    "ARCHIVED", "EXPORTED", "SYNCED_WITH", "BACKED_UP_TO", "MONITORS", "TRIGGERED",
    "RESOLVED", "ESCALATED", "AUDITED", "RENEWED", "SIGNED", "ISSUED", "REDEEMED",
]

PROPERTIES = [
    ("name", "STRING"), ("title", "STRING"), ("year", "INTEGER"), ("date", "DATE"),
    ("createdAt", "DATE_TIME"), ("score", "FLOAT"), ("rating", "FLOAT"),
    ("country", "STRING"), ("state", "STRING"), ("city", "STRING"), ("email", "STRING"),
    ("status", "STRING"), ("type", "STRING"), ("description", "STRING"),
    ("price", "FLOAT"), ("amount", "FLOAT"), ("count", "INTEGER"), ("code", "STRING"),
    ("url", "STRING"), ("language", "STRING"), ("duration", "INTEGER"),
    ("level", "INTEGER"), ("active", "BOOLEAN"), ("address", "STRING"),
]


def n_distractors(n_real, share):
    if share <= 0:
        return 0
    if share >= 1:
        raise ValueError("share must be < 1")
    return round(n_real * share / (1 - share))


def add_distractors(schema, share, seed=0, hard_fraction=0.5):
    """
    Return (noisy_schema, info). `info` has the distractor label / relationship
    names and counts. The input schema is not modified.
    """
    s = json.loads(schema) if isinstance(schema, str) else schema
    real_labels = sorted(set(s.get("node_props", {}))
                         | {r["start"] for r in s.get("relationships", [])}
                         | {r["end"] for r in s.get("relationships", [])})
    real_rels = sorted({r["type"] for r in s.get("relationships", [])})
    n_real = len(real_labels) + len(real_rels)
    n = n_distractors(n_real, share)
    n_lab = (n + 1) // 2
    n_rel = n - n_lab

    rnd = random.Random(f"{seed}:{share}:{','.join(real_labels)}")
    taken = set(real_labels)

    hard = [f"{L}{suf}" for L in real_labels for suf in HARD_SUFFIXES]
    easy = list(EASY_LABELS)
    rnd.shuffle(hard)
    rnd.shuffle(easy)
    n_hard = round(n_lab * hard_fraction)
    pool = [x for x in hard if x not in taken][:n_hard]
    pool += [x for x in easy if x not in taken and x not in pool]
    # if the pools run out, number them
    i = 2
    while len(pool) < n_lab:
        pool.append(f"{easy[(i - 2) % len(easy)]}{i}")
        i += 1
    d_labels = pool[:n_lab]

    node_props = {k: list(v) for k, v in s.get("node_props", {}).items()}
    for L in d_labels:
        k = rnd.randint(2, 5)
        node_props[L] = [{"property": p, "datatype": t} for p, t in rnd.sample(PROPERTIES, k)]

    verbs = [v for v in REL_VERBS if v not in real_rels]
    rnd.shuffle(verbs)
    j = 2
    while len(verbs) < n_rel:
        verbs.append(f"{REL_VERBS[(j - 2) % len(REL_VERBS)]}_{j}")
        j += 1
    d_rels = verbs[:n_rel]

    relationships = list(s.get("relationships", []))
    rel_props = {k: list(v) for k, v in s.get("rel_props", {}).items()}
    for T in d_rels:
        # always touch a distractor label; half the time connect to a real one
        a = rnd.choice(d_labels) if d_labels else rnd.choice(real_labels)
        b = rnd.choice(real_labels if (rnd.random() < 0.5 or len(d_labels) < 2) else d_labels)
        start, end = (a, b) if rnd.random() < 0.5 else (b, a)
        relationships.append({"start": start, "type": T, "end": end})
        if rnd.random() < 0.3:
            p, t = rnd.choice(PROPERTIES)
            rel_props[T] = [{"property": p, "datatype": t}]

    noisy = {"node_props": node_props, "rel_props": rel_props,
             "relationships": relationships}
    return noisy, {"share": share, "seed": seed, "n_real": n_real,
                   "n_distractors": len(d_labels) + len(d_rels),
                   "labels": d_labels, "rels": d_rels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--data-source", default="er")
    p.add_argument("--share", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()

    import pandas as pd
    df = pd.read_parquet(a.parquet)
    schema = df[df["data_source"] == a.data_source]["schema"].iloc[0]
    noisy, info = add_distractors(schema, a.share, a.seed)
    print(f"share={a.share}: {info['n_real']} real + {info['n_distractors']} distractor "
          f"elements; schema text {len(schema)} -> {len(json.dumps(noisy))} chars")
    print("distractor labels:", ", ".join(info["labels"][:15]),
          "..." if len(info["labels"]) > 15 else "")
    print("distractor rels:  ", ", ".join(info["rels"][:15]),
          "..." if len(info["rels"]) > 15 else "")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(noisy, f, indent=2)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
