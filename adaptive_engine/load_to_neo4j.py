#!/usr/bin/env python3
# load_to_neo4j.py
#
# Load questions + interactions CSVs into a local Neo4j instance.
#
# Creates a knowledge graph:
#   (q:Question {id, difficulty})-[:IN_TERM]->(t:Term {name})-[:IN_CATEGORY]->(c:Category {name})
# and student interactions:
#   (s:Student {id})-[:ATTEMPTED {ts, correctness}]->(q)
#
# Optionally computes and writes "MASTERY" edges:
#   (s)-[:MASTERY {mu, n, last_ts}]->(t)
# using a simple leakage-free sequential EWMA update per student-term.
#
# Usage:
#   python load_to_neo4j.py \
#     --questions questions.csv \
#     --interactions interactions.csv \
#     --neo4j-uri bolt://localhost:7687 \
#     --neo4j-user neo4j \
#     --neo4j-pass password \
#     --create-mastery
#
# Notes:
# - Expects questions.csv columns: question_id, category, difficulty, term
# - Expects interactions.csv columns: student_id, question_id, correctness, timestamp
# - Stores ATTEMPTED.ts as epoch seconds (int), correctness as float in [0,1]
# - Designed to run against a local Neo4j (e.g. docker run neo4j:5 ...)

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


# -------------------------
# Helpers
# -------------------------

def to_epoch_seconds(ts_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts_series, errors="coerce", utc=True)
    if ts.isna().all():
        # fallback: numeric seconds
        ts = pd.to_datetime(pd.to_numeric(ts_series, errors="coerce"), unit="s", utc=True)
    ts = ts.dropna()
    # pandas datetime64[ns, UTC] -> int64 ns -> seconds
    return (ts.astype("int64") // 10**9).astype("int64")


def chunked(lst: List[dict], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# -------------------------
# Cypher: schema
# -------------------------

CYPHER_CONSTRAINTS = [
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
    "CREATE CONSTRAINT student_id IF NOT EXISTS FOR (s:Student) REQUIRE s.id IS UNIQUE",
]

# Optional indexes (helpful for perf; Neo4j 5 syntax supports CREATE INDEX IF NOT EXISTS)
CYPHER_INDEXES = [
    "CREATE INDEX question_difficulty IF NOT EXISTS FOR (q:Question) ON (q.difficulty)",
]


def create_schema(driver):
    with driver.session() as sess:
        for stmt in CYPHER_CONSTRAINTS:
            sess.run(stmt)
        for stmt in CYPHER_INDEXES:
            sess.run(stmt)


# -------------------------
# Cypher: loaders
# -------------------------

CYPHER_LOAD_QUESTIONS = """
UNWIND $rows AS row
MERGE (c:Category {name: row.category})
MERGE (t:Term {name: row.term})
MERGE (q:Question {id: row.question_id})
SET q.difficulty = row.difficulty
MERGE (t)-[:IN_CATEGORY]->(c)
MERGE (q)-[:IN_TERM]->(t)
"""

CYPHER_LOAD_INTERACTIONS = """
UNWIND $rows AS row
MERGE (s:Student {id: row.student_id})
MERGE (q:Question {id: row.question_id})
MERGE (s)-[a:ATTEMPTED {ts: row.ts}]->(q)
SET a.correctness = row.correctness
"""

# Upsert mastery edge with provided values (computed in Python)
CYPHER_UPSERT_MASTERY = """
UNWIND $rows AS row
MERGE (s:Student {id: row.student_id})
MERGE (t:Term {name: row.term})
MERGE (s)-[m:MASTERY]->(t)
SET m.mu = row.mu,
    m.n = row.n,
    m.last_ts = row.last_ts
"""


# -------------------------
# Loading functions
# -------------------------

def load_questions(driver, questions_df: pd.DataFrame, batch_size: int):
    req_cols = {"question_id", "category", "difficulty", "term"}
    missing = req_cols - set(questions_df.columns)
    if missing:
        raise ValueError(f"Questions CSV missing columns: {missing}")

    df = questions_df.copy()
    df["question_id"] = df["question_id"].astype(str)
    df["category"] = df["category"].astype(str)
    df["difficulty"] = df["difficulty"].astype(str)
    df["term"] = df["term"].astype(str)

    rows = df[["question_id", "category", "difficulty", "term"]].to_dict("records")

    with driver.session() as sess:
        for part in chunked(rows, batch_size):
            sess.run(CYPHER_LOAD_QUESTIONS, rows=part)


def load_interactions(driver, interactions_df: pd.DataFrame, batch_size: int):
    req_cols = {"student_id", "question_id", "correctness", "timestamp"}
    missing = req_cols - set(interactions_df.columns)
    if missing:
        raise ValueError(f"Interactions CSV missing columns: {missing}")

    df = interactions_df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    # Convert timestamps -> epoch seconds
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.isna().all():
        # fallback numeric seconds
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["ts"] = (df["_ts"].astype("int64") // 10**9).astype(np.int64)

    rows = df[["student_id", "question_id", "correctness", "ts"]].to_dict("records")

    with driver.session() as sess:
        for part in chunked(rows, batch_size):
            sess.run(CYPHER_LOAD_INTERACTIONS, rows=part)


# -------------------------
# Mastery computation (optional)
# -------------------------

def compute_mastery_edges(
    questions_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    alpha: float,
) -> List[dict]:
    """
    Compute per (student, term) mastery summary:
      mu: EWMA of correctness in [0,1]
      n:  count of attempts in this term
      last_ts: last interaction ts (epoch seconds)

    EWMA is updated in chronological order per student.
    """
    qcols = {"question_id", "term"}
    if not qcols.issubset(set(questions_df.columns)):
        raise ValueError(f"Questions CSV must include {qcols} to compute mastery")

    icols = {"student_id", "question_id", "correctness", "timestamp"}
    if not icols.issubset(set(interactions_df.columns)):
        raise ValueError(f"Interactions CSV must include {icols} to compute mastery")

    qmap = questions_df[["question_id", "term"]].copy()
    qmap["question_id"] = qmap["question_id"].astype(str)
    qmap["term"] = qmap["term"].astype(str)
    q_to_term = dict(zip(qmap["question_id"], qmap["term"]))

    df = interactions_df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(np.float32)

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["ts"] = (df["_ts"].astype("int64") // 10**9).astype(np.int64)

    # attach term
    df["term"] = df["question_id"].map(q_to_term)
    df = df.dropna(subset=["term"])

    df = df.sort_values(["student_id", "ts"])

    # mastery state per student-term
    mastery: Dict[Tuple[str, str], Tuple[float, int, int]] = {}  # (sid, term) -> (mu, n, last_ts)
    for sid, g in df.groupby("student_id", sort=False):
        for _, r in g.iterrows():
            term = str(r["term"])
            y = float(r["correctness"])
            ts_i = int(r["ts"])
            key = (sid, term)
            if key in mastery:
                mu, n, _last = mastery[key]
                mu = (1.0 - alpha) * mu + alpha * y
                n = n + 1
                mastery[key] = (float(mu), int(n), ts_i)
            else:
                mastery[key] = (float(y), 1, ts_i)

    rows = []
    for (sid, term), (mu, n, last_ts) in mastery.items():
        rows.append(
            {
                "student_id": sid,
                "term": term,
                "mu": float(mu),
                "n": int(n),
                "last_ts": int(last_ts),
            }
        )
    return rows


def upsert_mastery(driver, mastery_rows: List[dict], batch_size: int):
    with driver.session() as sess:
        for part in chunked(mastery_rows, batch_size):
            sess.run(CYPHER_UPSERT_MASTERY, rows=part)


# -------------------------
# Optional: wipe DB (dangerous)
# -------------------------

def wipe_database(driver):
    with driver.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Path to questions CSV")
    ap.add_argument("--interactions", required=True, help="Path to interactions CSV")

    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", required=True)

    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--wipe", action="store_true", help="DANGEROUS: delete all nodes/edges before loading")

    ap.add_argument("--create-mastery", action="store_true", help="Compute & write (Student)-[:MASTERY]->(Term)")
    ap.add_argument("--alpha", type=float, default=0.2, help="EWMA smoothing for mastery mu")
    args = ap.parse_args()

    questions_df = pd.read_csv(args.questions)
    interactions_df = pd.read_csv(args.interactions)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))

    if args.wipe:
        print("Wiping database...")
        wipe_database(driver)

    print("Creating schema (constraints/indexes)...")
    create_schema(driver)

    print("Loading questions KG...")
    load_questions(driver, questions_df, batch_size=args.batch_size)

    print("Loading student interactions...")
    load_interactions(driver, interactions_df, batch_size=args.batch_size)

    if args.create_mastery:
        print("Computing mastery edges (EWMA, leakage-free chronological updates)...")
        mastery_rows = compute_mastery_edges(questions_df, interactions_df, alpha=args.alpha)
        print(f"Upserting mastery edges: {len(mastery_rows):,}")
        upsert_mastery(driver, mastery_rows, batch_size=args.batch_size)

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()