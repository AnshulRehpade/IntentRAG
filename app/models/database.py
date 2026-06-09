"""
PostgreSQL database models — SQLAlchemy ORM.

Tables:
- tenants
- users
- query_logs
- documents
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    tenant_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="tenant")
    documents = relationship("Document", back_populates="tenant")
    query_logs = relationship("QueryLog", back_populates="tenant")


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.tenant_id"), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="reader")  # admin | reader | writer
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="users")


class QueryLog(Base):
    __tablename__ = "query_logs"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.tenant_id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    query_text = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True)
    retrieved_chunks = Column(Text, nullable=True)  # JSON string
    llm_response = Column(Text, nullable=True)
    hallucination_flag = Column(Boolean, default=False)
    ragas_score = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="query_logs")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.tenant_id"), nullable=False)
    intent_category = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    embedding_id = Column(String(255), nullable=True)  # Reference to Qdrant point ID
    source = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="documents")
