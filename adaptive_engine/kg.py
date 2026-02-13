import math
from typing import Dict, List, Tuple, Optional

# ---------------------------
# Neo4j queries
# ---------------------------


def neo4j_fetch_history(
    driver,
    student_id: str,
    limit: int = 200,
) -> List[Tuple[str, float, int]]:
    """
    Returns a list of (question_id, correctness, ts_sec), sorted ascending by ts.
    """
    cypher = """
    MATCH (s:Student {id:$sid})-[a:ATTEMPTED]->(q:Question)
    RETURN q.id AS qid, a.correctness AS corr, a.ts AS ts
    ORDER BY a.ts ASC
    """
    # If ts is stored as datetime, replace a.ts with datetime(a.ts).epochSeconds or similar in your ingest.
    with driver.session() as sess:
        rows = sess.run(cypher, sid=student_id).data()

    # Robustness: keep last 'limit'
    hist = []
    for r in rows[-limit:]:
        qid = str(r["qid"])
        corr = float(r["corr"]) if r["corr"] is not None else 0.0
        ts = int(r["ts"]) if r["ts"] is not None else 0
        hist.append((qid, corr, ts))
    return hist


def neo4j_fetch_term_mastery(
    driver,
    student_id: str,
) -> Dict[str, Tuple[float, int, int]]:
    """
    Returns dict term_name -> (mu, n, last_ts_sec)
    """
    cypher = """
    MATCH (s:Student {id:$sid})-[m:MASTERY]->(t:Term)
    RETURN t.name AS term, m.mu AS mu, m.n AS n, m.last_ts AS last_ts
    """
    with driver.session() as sess:
        rows = sess.run(cypher, sid=student_id).data()

    out = {}
    for r in rows:
        term = str(r["term"])
        mu = float(r["mu"]) if r["mu"] is not None else 0.5
        n = int(r["n"]) if r["n"] is not None else 0
        last_ts = int(r["last_ts"]) if r["last_ts"] is not None else 0
        out[term] = (mu, n, last_ts)
    return out


def neo4j_pick_target_terms(
    driver,
    student_id: str,
    k: int = 5,
) -> List[Tuple[str, float, int, int]]:
    """
    Returns list of (term_name, mu, n, last_ts) for top-k terms by a simple priority.
    Falls back to all terms (no mastery) if none exist.
    """
    cypher_mastery = """
    MATCH (s:Student {id:$sid})-[m:MASTERY]->(t:Term)
    RETURN t.name AS term, m.mu AS mu, m.n AS n, m.last_ts AS last_ts
    """
    with driver.session() as sess:
        rows = sess.run(cypher_mastery, sid=student_id).data()

    if not rows:
        # fallback: suggest terms not yet covered (just return some terms)
        cypher_terms = """
        MATCH (t:Term)
        RETURN t.name AS term
        LIMIT $k
        """
        with driver.session() as sess:
            rows2 = sess.run(cypher_terms, k=k).data()
        return [(str(r["term"]), 0.5, 0, 0) for r in rows2]

    # Compute priority in python for clarity:
    items = []
    for r in rows:
        term = str(r["term"])
        mu = float(r["mu"]) if r["mu"] is not None else 0.5
        n = int(r["n"]) if r["n"] is not None else 0
        last_ts = int(r["last_ts"]) if r["last_ts"] is not None else 0
        # prioritize low mastery + low evidence
        priority = (1.0 - mu) + 0.4 / math.sqrt(n + 1.0)
        items.append((priority, term, mu, n, last_ts))

    items.sort(reverse=True, key=lambda x: x[0])
    return [(term, mu, n, last_ts) for _, term, mu, n, last_ts in items[:k]]


def neo4j_fetch_candidates(
    driver,
    student_id: str,
    terms: List[str],
    allowed_difficulties: Optional[List[str]] = None,
    limit: int = 500,
    exclude_seen: bool = True,
) -> List[Dict[str, str]]:
    """
    Returns list of candidate dicts:
      {qid, term, category, difficulty}
    """
    where_diff = ""
    params = {"sid": student_id, "terms": terms, "limit": limit}
    if allowed_difficulties:
        where_diff = "AND q.difficulty IN $diffs"
        params["diffs"] = allowed_difficulties

    exclude_clause = ""
    if exclude_seen:
        exclude_clause = """
        AND NOT EXISTS {
          MATCH (s:Student {id:$sid})-[:ATTEMPTED]->(q)
        }
        """

    cypher = f"""
    MATCH (t:Term)<-[:IN_TERM]-(q:Question)
    MATCH (t)-[:IN_CATEGORY]->(c:Category)
    WHERE t.name IN $terms
    {where_diff}
    {exclude_clause}
    RETURN q.id AS qid, t.name AS term, c.name AS category, q.difficulty AS difficulty
    LIMIT $limit
    """

    with driver.session() as sess:
        rows = sess.run(cypher, **params).data()

    cands = []
    for r in rows:
        cands.append(
            {
                "qid": str(r["qid"]),
                "term": str(r["term"]),
                "category": str(r["category"]),
                "difficulty": str(r["difficulty"]),
            }
        )
    return cands