"""
Database models for storing experiments, trials, conversations, and messages.

This module defines SQLAlchemy models for persisting red teaming experiment data,
including the Message objects from reveal.core.shared.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (JSON, DateTime, ForeignKey, Integer, LargeBinary,
                        PickleType, String, Text, select)
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from reveal.core import shared
from reveal.settings import Settings


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    description: Mapped[str] = mapped_column(String)
    base_prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class Trial(Base):
    __tablename__ = "trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    system_prompt: Mapped[str] = mapped_column(Text)
    tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class Conversation(Base):
    """Groups messages together into a conversation"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id"), index=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)

    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mutation_applied: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class Message(Base):
    """Stores Message objects from reveal.core.shared"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id"), index=True)
    conversation_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("conversations.id"), index=True, nullable=True
    )
    sequence: Mapped[int] = mapped_column(Integer)  # Order in trial

    # Core Message fields
    role: Mapped[str] = mapped_column(String)  # user/assistant/system/tool
    type: Mapped[str] = mapped_column(
        String
    )  # text/reasoning/structured_output/function_call/function_output
    name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # Persona or tool name

    # Content fields (only one will be populated based on type)
    content: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # For text messages
    reasoning: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # For reasoning messages
    structured_output: Mapped[Optional[object]] = mapped_column(
        PickleType, nullable=True
    )  # For structured output
    function_call: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # {"name": str, "arguments": str}
    function_output: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Tool execution result

    _raw: Mapped[Optional[object]] = mapped_column(PickleType, nullable=True)

    # Model info (for assistant messages)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(ForeignKey("trials.id"))

    violation_type: Mapped[str] = mapped_column(String)
    reasoning: Mapped[str] = mapped_column(Text)


# Database setup
engine = create_async_engine(
    f"sqlite+aiosqlite:///{Settings.data_dir}/experiments.db",
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_trial_messages(
    session: AsyncSession,
    trial_id: int,
    messages: list[shared.Message],
) -> list[int]:
    """Save a list of Message objects to the database and return their IDs"""

    message_ids = []
    for i, msg in enumerate(messages):
        db_msg = Message(
            trial_id=trial_id,
            sequence=i,
            role=msg.role,
            type=msg.type,
            name=msg.name,
            content=msg.content,
            reasoning=msg.reasoning,
            structured_output=msg.structured_output if msg.structured_output else None,
            function_call=msg.function_call,
            function_output=msg.function_output,
            _raw=msg._raw,
            model=getattr(msg._raw, "model", None) if msg._raw else None,
        )
        session.add(db_msg)
        await session.flush()  # Flush to get the ID
        message_ids.append(db_msg.id)

    await session.commit()
    return message_ids


async def get_trial_and_messages_from_violation(
    violation_id: int,
) -> Optional[tuple[Trial, list[Message]]]:
    async with AsyncSessionLocal() as session:
        violation_stmt = select(Violation).where(Violation.id == violation_id)
        violation_result = await session.execute(violation_stmt)
        violation = violation_result.scalar_one_or_none()
        
        if violation is None:
            return None
            
        trial_stmt = select(Trial).where(Trial.id == violation.trial_id)
        trial_result = await session.execute(trial_stmt)
        trial = trial_result.scalar_one_or_none()
        
        if trial is None:
            return None
            
        messages_stmt = select(Message).where(
            Message.trial_id == violation.trial_id
        ).order_by(Message.sequence)
        messages_result = await session.execute(messages_stmt)
        messages = messages_result.scalars().all()
        
        return (trial, list(messages))
