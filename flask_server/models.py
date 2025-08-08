# ./flask_server/models.py
from datetime import datetime
from typing import Optional, List

import sqlalchemy as sa
from sqlalchemy import func
import sqlalchemy.orm as so
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from flask_server import db


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

    def __repr__(self):
        return (
            f"User(id={self.id}, username={self.username},"
            f" email={self.email}, role={self.role})"
        )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


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
        sa.DateTime, server_default=func.now()
    )

    user: so.Mapped[User] = so.relationship("User", back_populates="images")

    def __repr__(self):
        return (
            f"GeneratedImage(id={self.id}, user_id={self.user_id},"
            f" filename={self.filename})"
        )


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
        sa.DateTime, server_default=func.now()
    )

    user: so.Mapped[User] = so.relationship("User", back_populates="models")

    def __repr__(self):
        return (
            f"GeneratedModel(id={self.id}, user_id={self.user_id},"
            f" filename={self.filename})"
        )
