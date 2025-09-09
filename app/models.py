from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class MetricRecord(SQLModel, table=True):
    """
    Simplified model for storing metric records.

    Contains only the core fields: project, model, dataset, metric, value, timestamp.
    Designed to be minimal and extensible for future enhancements.
    """

    __tablename__ = "metric_records"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    project: str = Field(max_length=100, description="Project name")
    model: str = Field(max_length=100, description="Model name")
    dataset: str = Field(max_length=100, description="Dataset name")
    metric: str = Field(max_length=50, description="Metric name")
    value: float = Field(description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the record was created")


class MetricRecordCreate(SQLModel, table=False):
    """Schema for creating new metric records."""

    project: str = Field(max_length=100)
    model: str = Field(max_length=100)
    dataset: str = Field(max_length=100)
    metric: str = Field(max_length=50)
    value: float


class MetricRecordUpdate(SQLModel, table=False):
    """Schema for updating metric records."""

    project: Optional[str] = Field(default=None, max_length=100)
    model: Optional[str] = Field(default=None, max_length=100)
    dataset: Optional[str] = Field(default=None, max_length=100)
    metric: Optional[str] = Field(default=None, max_length=50)
    value: Optional[float] = Field(default=None)
