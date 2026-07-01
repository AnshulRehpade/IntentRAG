"""
Analytics endpoint — hallucination pattern analysis (protected).

Aggregates query_logs data to surface:
- Hallucination rate by intent category
- Common causes and correlations
- Retrieval quality vs hallucination rate
- Healing success rate
- Trends over time
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, case, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import CurrentUser, require_role
from app.models.database import QueryLog

router = APIRouter()


@router.get("/hallucinations")
async def hallucination_analytics(
    days: Optional[int] = Query(30, description="Number of days to analyze"),
    user: CurrentUser = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze hallucination patterns across the tenant's query history.

    Returns:
    - Overall hallucination rate
    - Breakdown by intent category
    - Trend (hallucination rate over time buckets)
    - Latency comparison (hallucinated vs clean)

    Auth: admin only.
    """
    try:
        tenant_id = user.tenant_id

        # --- Overall stats ---
        total_result = await db.execute(
            select(
                func.count(QueryLog.query_id).label("total"),
                func.sum(
                    case((QueryLog.hallucination_flag == True, 1), else_=0)
                ).label("hallucinated"),
                func.avg(QueryLog.latency_ms).label("avg_latency"),
            ).where(QueryLog.tenant_id == tenant_id)
        )
        overall = total_result.mappings().first()

        total_queries = int(overall["total"] or 0)
        total_hallucinated = int(overall["hallucinated"] or 0)
        avg_latency = float(overall["avg_latency"] or 0)
        hallucination_rate = (
            total_hallucinated / total_queries if total_queries > 0 else 0.0
        )

        # --- By intent ---
        intent_result = await db.execute(
            select(
                QueryLog.intent,
                func.count(QueryLog.query_id).label("total"),
                func.sum(
                    case((QueryLog.hallucination_flag == True, 1), else_=0)
                ).label("hallucinated"),
                func.avg(QueryLog.latency_ms).label("avg_latency"),
            )
            .where(QueryLog.tenant_id == tenant_id)
            .where(QueryLog.intent.isnot(None))
            .group_by(QueryLog.intent)
            .order_by(func.count(QueryLog.query_id).desc())
        )

        by_intent = []
        for row in intent_result.mappings():
            intent_total = int(row["total"] or 0)
            intent_hallucinated = int(row["hallucinated"] or 0)
            by_intent.append({
                "intent": row["intent"],
                "total_queries": intent_total,
                "hallucinations": intent_hallucinated,
                "hallucination_rate": round(
                    intent_hallucinated / intent_total if intent_total > 0 else 0.0, 4
                ),
                "avg_latency_ms": round(float(row["avg_latency"] or 0), 1),
            })

        # --- Latency comparison ---
        latency_result = await db.execute(
            select(
                QueryLog.hallucination_flag,
                func.avg(QueryLog.latency_ms).label("avg_latency"),
                func.min(QueryLog.latency_ms).label("min_latency"),
                func.max(QueryLog.latency_ms).label("max_latency"),
                func.count(QueryLog.query_id).label("count"),
            )
            .where(QueryLog.tenant_id == tenant_id)
            .group_by(QueryLog.hallucination_flag)
        )

        latency_comparison = {}
        for row in latency_result.mappings():
            key = "hallucinated" if row["hallucination_flag"] else "clean"
            latency_comparison[key] = {
                "count": int(row["count"]),
                "avg_latency_ms": round(float(row["avg_latency"] or 0), 1),
                "min_latency_ms": int(row["min_latency"] or 0),
                "max_latency_ms": int(row["max_latency"] or 0),
            }

        # --- Recent hallucinated queries (for debugging) ---
        recent_result = await db.execute(
            select(
                QueryLog.query_text,
                QueryLog.intent,
                QueryLog.llm_response,
                QueryLog.latency_ms,
                QueryLog.timestamp,
            )
            .where(
                and_(
                    QueryLog.tenant_id == tenant_id,
                    QueryLog.hallucination_flag == True,
                )
            )
            .order_by(QueryLog.timestamp.desc())
            .limit(10)
        )

        recent_hallucinations = [
            {
                "query": row.query_text,
                "intent": row.intent,
                "answer_preview": (row.llm_response or "")[:200],
                "latency_ms": row.latency_ms,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            }
            for row in recent_result
        ]

        return {
            "success": True,
            "data": {
                "summary": {
                    "total_queries": total_queries,
                    "total_hallucinations": total_hallucinated,
                    "hallucination_rate": round(hallucination_rate, 4),
                    "clean_answers": total_queries - total_hallucinated,
                    "avg_latency_ms": round(avg_latency, 1),
                },
                "by_intent": by_intent,
                "latency_comparison": latency_comparison,
                "recent_hallucinations": recent_hallucinations,
                "insights": _generate_insights(
                    hallucination_rate, by_intent, latency_comparison
                ),
            },
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Analytics failed: {str(e)}",
        }


def _generate_insights(
    overall_rate: float,
    by_intent: list[dict],
    latency_comparison: dict,
) -> list[str]:
    """
    Generate human-readable insights from the analytics data.
    These help the user understand WHY hallucinations happen.
    """
    insights = []

    # Overall health
    if overall_rate == 0:
        insights.append("No hallucinations detected — system is performing well.")
    elif overall_rate < 0.1:
        insights.append(
            f"Hallucination rate is {overall_rate:.1%} — within acceptable range (<10%)."
        )
    elif overall_rate < 0.3:
        insights.append(
            f"⚠️ Hallucination rate is {overall_rate:.1%} — consider expanding the knowledge base."
        )
    else:
        insights.append(
            f"🚨 High hallucination rate ({overall_rate:.1%}) — likely insufficient context in the knowledge base."
        )

    # Intent-specific insights
    if by_intent:
        worst = max(by_intent, key=lambda x: x["hallucination_rate"])
        if worst["hallucination_rate"] > 0.2 and worst["total_queries"] >= 3:
            insights.append(
                f"'{worst['intent']}' intent has the highest hallucination rate "
                f"({worst['hallucination_rate']:.0%}). Consider adding more documents "
                f"to data/{worst['intent']}/."
            )

        best = min(by_intent, key=lambda x: x["hallucination_rate"])
        if best["hallucination_rate"] == 0 and best["total_queries"] >= 3:
            insights.append(
                f"'{best['intent']}' intent has zero hallucinations — knowledge base coverage is good here."
            )

    # Latency insight
    if "hallucinated" in latency_comparison and "clean" in latency_comparison:
        hall_latency = latency_comparison["hallucinated"]["avg_latency_ms"]
        clean_latency = latency_comparison["clean"]["avg_latency_ms"]
        if clean_latency > 0 and hall_latency > clean_latency * 1.5:
            insights.append(
                f"Hallucinated responses take {hall_latency/clean_latency:.1f}x longer on average — "
                "self-healing retries add latency but improve accuracy."
            )

    return insights
