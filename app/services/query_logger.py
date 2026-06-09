"""
Query logging service — persists query metadata to PostgreSQL.
"""

import json
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import QueryLog


class QueryLogger:
    """
    Logs every query through the pipeline to the query_logs table.
    Used for analytics, debugging, and audit trails.
    """

    @staticmethod
    async def log(
        db: AsyncSession,
        tenant_id: str,
        user_id: Optional[str],
        query_text: str,
        intent: Optional[str] = None,
        retrieved_chunks: Optional[list] = None,
        llm_response: Optional[str] = None,
        hallucination_flag: bool = False,
        ragas_score: Optional[float] = None,
        latency_ms: Optional[int] = None,
    ) -> str:
        """
        Log a query execution to the database.

        Returns the query_id as a string.
        """
        log_entry = QueryLog(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=query_text,
            intent=intent,
            retrieved_chunks=json.dumps(retrieved_chunks) if retrieved_chunks else None,
            llm_response=llm_response,
            hallucination_flag=hallucination_flag,
            ragas_score=ragas_score,
            latency_ms=latency_ms,
        )
        db.add(log_entry)
        await db.commit()
        await db.refresh(log_entry)
        return str(log_entry.query_id)
