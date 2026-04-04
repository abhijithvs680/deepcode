"""Ingest-time code intelligence: symbols, rich metadata, import graph, Git per file."""
import uuid

from sqlalchemy import Boolean, Column, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class RepoSymbol(Base):
    """Maps a defined symbol (class/function/module alias) to a source file."""

    __tablename__ = "repo_symbols"
    __table_args__ = (
        UniqueConstraint("repo_name", "symbol", "file_path", name="uq_repo_symbol_file"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    kind = Column(String(32), nullable=False, default="name")


class RepoSymbolMeta(Base):
    """Rich symbol metadata (methods, signatures, docstrings) for retrieval and context."""

    __tablename__ = "repo_symbol_meta"
    __table_args__ = (
        UniqueConstraint(
            "repo_name", "file_path", "qualified_name",
            name="uq_repo_symbol_meta_qualified",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False, index=True)
    qualified_name = Column(String, nullable=False, index=True)
    kind = Column(String(32), nullable=False)
    parent_symbol = Column(String, nullable=True)
    line_start = Column(Integer, nullable=False)
    line_end = Column(Integer, nullable=False)
    bases_json = Column(Text, nullable=True)
    signature = Column(Text, nullable=True)
    return_annotation = Column(Text, nullable=True)
    docstring = Column(Text, nullable=True)
    is_async = Column(Boolean, nullable=False, default=False)
    is_classmethod = Column(Boolean, nullable=False, default=False)
    is_staticmethod = Column(Boolean, nullable=False, default=False)


class RepoImportEdge(Base):
    """Directed edge: source_file imports target_file (resolved static path)."""

    __tablename__ = "repo_import_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    source_file = Column(String, nullable=False, index=True)
    target_file = Column(String, nullable=False, index=True)


class RepoFileGit(Base):
    """Per-file Git summary for scoring and context (ingest snapshot)."""

    __tablename__ = "repo_file_git"
    __table_args__ = (
        UniqueConstraint("repo_name", "file_path", name="uq_repo_file_git_path"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False, index=True)
    last_commit_hash = Column(String(64), nullable=True)
    last_commit_time = Column(String(64), nullable=True)
    last_author = Column(String(256), nullable=True)
    last_message = Column(Text, nullable=True)
    recent_commits_json = Column(Text, nullable=True)
    blame_summary_json = Column(Text, nullable=True)
