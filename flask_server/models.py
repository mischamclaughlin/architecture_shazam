# ./flask_server/models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import sqlalchemy as sa
import sqlalchemy.orm as so
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from flask_server import db


# -------------------- Users --------------------
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    username: so.Mapped[str] = so.mapped_column(sa.String(64), index=True, unique=True)
    email: so.Mapped[str] = so.mapped_column(sa.String(120), index=True, unique=True)
    password_hash: so.Mapped[Optional[str]] = so.mapped_column(sa.String(256))
    role: so.Mapped[str] = so.mapped_column(
        sa.String(24), nullable=False, default="Normal"
    )

    images: so.Mapped[List["GeneratedImage"]] = so.relationship(
        "GeneratedImage", back_populates="user", cascade="all, delete-orphan"
    )
    models: so.Mapped[List["GeneratedModel"]] = so.relationship(
        "GeneratedModel", back_populates="user", cascade="all, delete-orphan"
    )
    analyses: so.Mapped[List["Analysis"]] = so.relationship(
        "Analysis", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"User(id={self.id}, username={self.username}, email={self.email}, role={self.role})"

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


# -------------------- Stored assets --------------------
class GeneratedImage(db.Model):
    __tablename__ = "generated_images"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    user_id: so.Mapped[int] = so.mapped_column(
        sa.ForeignKey("users.id"), nullable=False
    )
    filename: so.Mapped[str] = so.mapped_column(sa.String(256), nullable=False)
    mime_type: so.Mapped[str] = so.mapped_column(sa.String(100), nullable=False)
    data: so.Mapped[bytes] = so.mapped_column(sa.LargeBinary, nullable=False)
    created_at: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )

    user: so.Mapped[User] = so.relationship("User", back_populates="images")

    def __repr__(self):
        return f"GeneratedImage(id={self.id}, user_id={self.user_id}, filename={self.filename})"


class GeneratedModel(db.Model):
    __tablename__ = "generated_models"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    user_id: so.Mapped[int] = so.mapped_column(
        sa.ForeignKey("users.id"), nullable=False
    )
    filename: so.Mapped[str] = so.mapped_column(sa.String(256), nullable=False)
    mime_type: so.Mapped[str] = so.mapped_column(sa.String(100), nullable=False)
    data: so.Mapped[bytes] = so.mapped_column(sa.LargeBinary, nullable=False)
    created_at: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )

    user: so.Mapped[User] = so.relationship("User", back_populates="models")

    def __repr__(self):
        return f"GeneratedModel(id={self.id}, user_id={self.user_id}, filename={self.filename})"


# -------------------- Pipeline tables --------------------
class Analysis(db.Model):
    __tablename__ = "analyses"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    user_id: so.Mapped[int] = so.mapped_column(
        sa.Integer, sa.ForeignKey("users.id"), nullable=False
    )

    title: so.Mapped[Optional[str]] = so.mapped_column(sa.String(255))
    artist: so.Mapped[Optional[str]] = so.mapped_column(sa.String(255))
    album: so.Mapped[Optional[str]] = so.mapped_column(sa.String(255))
    release: so.Mapped[Optional[str]] = so.mapped_column(sa.String(64))
    audio_path: so.Mapped[Optional[str]] = so.mapped_column(sa.String(1024))

    features: so.Mapped[Dict[str, Any]] = so.mapped_column(sa.JSON, nullable=False)
    genres: so.Mapped[List[Tuple[str, float]]] = so.mapped_column(
        sa.JSON, nullable=False
    )
    instruments: so.Mapped[List[Tuple[str, float]]] = so.mapped_column(
        sa.JSON, nullable=False
    )
    origin: so.Mapped[Tuple[str, str]] = so.mapped_column(sa.JSON, nullable=False)

    created_at: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )

    user: so.Mapped[User] = so.relationship("User", back_populates="analyses")
    prompts: so.Mapped[List["Prompt"]] = so.relationship(
        "Prompt", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Analysis(id={self.id}, user_id={self.user_id}, title={self.title})"


class Prompt(db.Model):
    __tablename__ = "prompts"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    analysis_id: so.Mapped[int] = so.mapped_column(
        sa.Integer, sa.ForeignKey("analyses.id"), nullable=False
    )

    building_type: so.Mapped[str] = so.mapped_column(sa.String(32), nullable=False)
    llm: so.Mapped[str] = so.mapped_column(sa.String(128), nullable=False)
    temperature: so.Mapped[float] = so.mapped_column(sa.Float, default=0.7)
    max_tokens: so.Mapped[int] = so.mapped_column(sa.Integer, default=10_000)

    raw_prompt: so.Mapped[str] = so.mapped_column(sa.Text, nullable=False)
    created_at: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )

    analysis: so.Mapped[Analysis] = so.relationship(
        "Analysis", back_populates="prompts"
    )
    render_tasks: so.Mapped[List["RenderTask"]] = so.relationship(
        "RenderTask", back_populates="prompt", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Prompt(id={self.id}, analysis_id={self.analysis_id}, building_type={self.building_type})"


class RenderTask(db.Model):
    __tablename__ = "render_tasks"

    id: so.Mapped[int] = so.mapped_column(sa.Integer, primary_key=True)
    prompt_id: so.Mapped[int] = so.mapped_column(
        sa.Integer, sa.ForeignKey("prompts.id"), nullable=False
    )

    action: so.Mapped[str] = so.mapped_column(sa.String(16), nullable=False)
    sdxl_prompt: so.Mapped[Optional[str]] = so.mapped_column(sa.Text)
    inference_steps: so.Mapped[Optional[int]] = so.mapped_column(sa.Integer)
    seed: so.Mapped[Optional[int]] = so.mapped_column(sa.Integer)

    status: so.Mapped[str] = so.mapped_column(
        sa.Enum("PENDING", "RUNNING", "SUCCEEDED", "FAILED", name="render_status"),
        nullable=False,
        default="PENDING",
    )
    error: so.Mapped[Optional[str]] = so.mapped_column(sa.Text)

    output_filename: so.Mapped[Optional[str]] = so.mapped_column(sa.String(512))
    created_at: so.Mapped[datetime] = so.mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    completed_at: so.Mapped[Optional[datetime]] = so.mapped_column(sa.DateTime)

    prompt: so.Mapped[Prompt] = so.relationship("Prompt", back_populates="render_tasks")

    def __repr__(self) -> str:
        return f"RenderTask(id={self.id}, prompt_id={self.prompt_id}, action={self.action}, status={self.status})"
