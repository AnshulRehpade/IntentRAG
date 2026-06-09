"""
Document ingestion endpoint — upload, chunk, embed, store (protected).
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import CurrentUser, require_role
from app.models.database import Document
from app.services.ingestion import ingestion_service

router = APIRouter()

# Allowed intent categories
VALID_INTENTS = {"factual", "person", "time", "location", "explanation", "other"}


@router.post("")
async def ingest_document(
    file: UploadFile = File(...),
    intent_category: str = Form(...),
    source: Optional[str] = Form(None),
    user: CurrentUser = Depends(require_role("admin", "writer")),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document, chunk it, generate embeddings, and store in Qdrant.

    - File content is read as UTF-8 text
    - Chunks stored in the intent_category collection
    - Each chunk carries tenant_id as metadata for isolation
    - Document record saved to PostgreSQL for tracking

    Auth: admin or writer only.
    """
    try:
        # Validate intent category
        if intent_category not in VALID_INTENTS:
            return {
                "success": False,
                "data": None,
                "error": f"Invalid intent_category '{intent_category}'. Must be one of: {VALID_INTENTS}",
            }

        # Read file content
        content = await file.read()
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "success": False,
                "data": None,
                "error": "File must be UTF-8 encoded text",
            }

        if not text_content.strip():
            return {
                "success": False,
                "data": None,
                "error": "File is empty",
            }

        # Save document record to PostgreSQL
        doc = Document(
            tenant_id=user.tenant_id,
            intent_category=intent_category,
            content=text_content[:5000],  # Store first 5k chars for reference
            source=source or file.filename,
        )
        db.add(doc)
        await db.flush()

        # Ingest into vector store
        result = await ingestion_service.ingest(
            content=text_content,
            intent_category=intent_category,
            tenant_id=user.tenant_id,
            source=source or file.filename,
            doc_id=str(doc.doc_id),
        )

        # Update document with embedding reference
        doc.embedding_id = ",".join(result["node_ids"][:5])  # Store first 5 node IDs
        await db.commit()

        return {
            "success": True,
            "data": {
                "doc_id": str(doc.doc_id),
                "tenant_id": user.tenant_id,
                "intent_category": intent_category,
                "filename": file.filename,
                "chunks_created": result["chunks_created"],
                "source": source or file.filename,
            },
            "error": None,
        }

    except Exception as e:
        await db.rollback()
        return {
            "success": False,
            "data": None,
            "error": f"Ingestion failed: {str(e)}",
        }
