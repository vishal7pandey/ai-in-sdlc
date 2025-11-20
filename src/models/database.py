"""SQLAlchemy ORM models for the Requirements Engineering Platform."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models.

    Using DeclarativeBase keeps mypy happy when referring to Base as a type
    and avoids "Base is not valid as a type" errors.
    """

    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_name = Column(String(255), nullable=False)
    user_id = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="active")
    version = Column(Integer, nullable=False, default=0)
    metadata_json = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    chat_messages = relationship(
        "ChatMessageModel", back_populates="session", cascade="all, delete-orphan"
    )
    requirements = relationship(
        "RequirementModel", back_populates="session", cascade="all, delete-orphan"
    )
    rd_documents = relationship("RDDocumentModel", back_populates="session")

    __table_args__ = (
        CheckConstraint("char_length(project_name) > 0", name="sessions_project_name_not_empty"),
        CheckConstraint(
            "status IN ('active', 'reviewing', 'approved', 'archived')",
            name="sessions_status_check",
        ),
        CheckConstraint("version >= 0", name="sessions_version_nonnegative"),
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_status", "status"),
        Index("idx_sessions_created_at", "created_at"),
    )


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata_json = Column(JSONB, nullable=False, default=dict)

    session = relationship("SessionModel", back_populates="chat_messages")

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="chat_role_check"),
        CheckConstraint("char_length(content) > 0", name="chat_message_content_not_empty"),
        Index("idx_chat_session_timestamp", "session_id", "timestamp"),
        Index("idx_chat_messages_role", "role"),
    )


class RequirementModel(Base):
    __tablename__ = "requirements"

    id = Column(String(50), primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    title = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)
    actor = Column(String(200), nullable=False)
    action = Column(Text, nullable=False)
    condition = Column(Text)
    acceptance_criteria = Column(JSONB, nullable=False)
    priority = Column(String(20), nullable=False)
    confidence = Column(Numeric(3, 2), nullable=False)
    inferred = Column(Boolean, default=False)
    rationale = Column(Text, nullable=False)
    source_refs = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    version = Column(Integer, default=0)

    session = relationship("SessionModel", back_populates="requirements")

    __table_args__ = (
        CheckConstraint(
            "id ~ '^REQ(-INF)?-\\d{3,}$'",
            name="requirements_id_pattern",
        ),
        CheckConstraint(
            "type IN ('functional','non-functional','business','security','data','interface','constraint')",
            name="requirements_type_check",
        ),
        CheckConstraint(
            "priority IN ('low','medium','high','must')", name="requirements_priority_check"
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="requirements_confidence"),
        CheckConstraint(
            "jsonb_array_length(acceptance_criteria) > 0",
            name="requirements_acceptance_not_empty",
        ),
        CheckConstraint("version >= 0", name="requirements_version_nonnegative"),
        Index("idx_requirements_session", "session_id"),
        Index("idx_requirements_type", "type"),
        Index("idx_requirements_priority", "priority"),
        Index("idx_requirements_inferred", "inferred"),
        Index("idx_requirements_created_at", "created_at"),
    )


class RDDocumentModel(Base):
    __tablename__ = "rd_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    version = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(20), default="markdown")
    status = Column(String(20), default="draft")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata_json = Column("metadata", JSONB, default=dict)

    session = relationship("SessionModel", back_populates="rd_documents")

    __table_args__ = (
        CheckConstraint("format IN ('markdown','json','pdf')", name="rd_format_check"),
        CheckConstraint("status IN ('draft','under_review','approved')", name="rd_status_check"),
        CheckConstraint("version > 0", name="rd_documents_version_positive"),
        UniqueConstraint("session_id", "version", name="uq_rd_documents_session_version"),
        Index("idx_rd_documents_session", "session_id"),
        Index("idx_rd_documents_version", "version"),
        Index("idx_rd_documents_session_version", "session_id", "version"),
    )


class RDEventModel(Base):
    __tablename__ = "rd_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    version = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSONB, nullable=False)
    user_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('created','updated','approved','exported')",
            name="rd_event_type_check",
        ),
        CheckConstraint("version > 0", name="rd_events_version_positive"),
        Index("idx_rd_events_session_version", "session_id", "version"),
        Index("idx_rd_events_session_timestamp", "session_id", "timestamp"),
    )


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"))
    user_id = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(100))
    changes = Column("changes", JSONB)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(INET)
    user_agent = Column(Text)

    __table_args__ = (
        Index("idx_audit_logs_session", "session_id", "timestamp"),
        Index("idx_audit_logs_user", "user_id", "timestamp"),
        Index("idx_audit_logs_entity", "entity_type", "entity_id"),
    )


class LangGraphCheckpointModel(Base):
    __tablename__ = "langgraph_checkpoints"

    thread_id = Column(String(255), primary_key=True)
    checkpoint_id = Column(UUID(as_uuid=True), primary_key=True)
    parent_checkpoint_id = Column(UUID(as_uuid=True))
    checkpoint = Column(JSONB, nullable=False)
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        PrimaryKeyConstraint("thread_id", "checkpoint_id"),
        Index("idx_checkpoint_thread_created", "thread_id", "created_at"),
    )


class LangGraphCheckpointWriteModel(Base):
    __tablename__ = "langgraph_checkpoint_writes"

    thread_id = Column(String(255), primary_key=True)
    checkpoint_id = Column(UUID(as_uuid=True), primary_key=True)
    task_id = Column(String(255), primary_key=True)
    idx = Column(Integer, primary_key=True)
    channel = Column(String(255), nullable=False)
    value = Column("value", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["thread_id", "checkpoint_id"],
            [
                "langgraph_checkpoints.thread_id",
                "langgraph_checkpoints.checkpoint_id",
            ],
            ondelete="CASCADE",
        ),
    )


MODEL_REGISTRY: tuple[type[Base], ...] = (
    SessionModel,
    ChatMessageModel,
    RequirementModel,
    RDEventModel,
    RDDocumentModel,
    AuditLogModel,
    LangGraphCheckpointModel,
    LangGraphCheckpointWriteModel,
)

__all__ = [m.__name__ for m in MODEL_REGISTRY] + ["Base"]
