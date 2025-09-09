from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal


class ExperimentMetric(SQLModel, table=True):
    """Model for storing machine learning experiment metrics."""

    __tablename__ = "experiment_metrics"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    project_name: str = Field(max_length=100, description="Name of the ML project")
    model_name: str = Field(max_length=100, description="Name of the machine learning model")
    dataset_name: str = Field(max_length=100, description="Name of the dataset used")
    metric_name: str = Field(max_length=50, description="Name of the metric (e.g., accuracy, loss, f1_score)")
    metric_value: Decimal = Field(description="Numerical value of the metric")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the metric was recorded")

    # Optional fields for extensibility
    notes: Optional[str] = Field(default=None, max_length=500, description="Additional notes about the experiment")
    experiment_id: Optional[str] = Field(
        default=None, max_length=100, description="External experiment ID for tracking"
    )


class ExperimentMetricCreate(SQLModel, table=False):
    """Schema for creating new experiment metrics."""

    project_name: str = Field(max_length=100)
    model_name: str = Field(max_length=100)
    dataset_name: str = Field(max_length=100)
    metric_name: str = Field(max_length=50)
    metric_value: Decimal
    notes: Optional[str] = Field(default=None, max_length=500)
    experiment_id: Optional[str] = Field(default=None, max_length=100)


class ExperimentMetricUpdate(SQLModel, table=False):
    """Schema for updating experiment metrics."""

    project_name: Optional[str] = Field(default=None, max_length=100)
    model_name: Optional[str] = Field(default=None, max_length=100)
    dataset_name: Optional[str] = Field(default=None, max_length=100)
    metric_name: Optional[str] = Field(default=None, max_length=50)
    metric_value: Optional[Decimal] = Field(default=None)
    notes: Optional[str] = Field(default=None, max_length=500)
    experiment_id: Optional[str] = Field(default=None, max_length=100)
