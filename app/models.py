from sqlmodel import SQLModel, Field, Relationship, JSON, Column
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum


# Enums for data types
class MetricType(str, Enum):
    """Types of metrics that can be tracked."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    DICE = "dice"
    IOU = "iou"
    LOSS = "loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    AUC = "auc"
    CUSTOM = "custom"


class ComparisonOperator(str, Enum):
    """Operators for metric target comparisons."""

    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="
    EQUAL = "=="


class ProjectStatus(str, Enum):
    """Project status options."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class ExperimentStatus(str, Enum):
    """Experiment status options."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# Persistent models (stored in database)
class Project(SQLModel, table=True):
    """Core project entity for organizing ML experiments."""

    __tablename__ = "projects"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, unique=True, index=True)
    description: str = Field(default="", max_length=1000)
    status: ProjectStatus = Field(default=ProjectStatus.ACTIVE)
    clearml_project_name: Optional[str] = Field(default=None, max_length=200)
    adapter_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    experiments: List["Experiment"] = Relationship(back_populates="project")
    metric_targets: List["MetricTarget"] = Relationship(back_populates="project")
    datasets: List["Dataset"] = Relationship(back_populates="project")


class Dataset(SQLModel, table=True):
    """Dataset entity for tracking data used in experiments."""

    __tablename__ = "datasets"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, index=True)
    description: str = Field(default="", max_length=1000)
    version: str = Field(default="1.0", max_length=50)
    project_id: int = Field(foreign_key="projects.id")
    clearml_dataset_id: Optional[str] = Field(default=None, max_length=200)
    extra_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    project: Project = Relationship(back_populates="datasets")
    experiments: List["Experiment"] = Relationship(back_populates="dataset")
    metric_points: List["MetricPoint"] = Relationship(back_populates="dataset")


class Model(SQLModel, table=True):
    """Model entity for tracking different ML models."""

    __tablename__ = "models"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, index=True)
    description: str = Field(default="", max_length=1000)
    architecture: str = Field(default="", max_length=200)
    version: str = Field(default="1.0", max_length=50)
    hyperparameters: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    experiments: List["Experiment"] = Relationship(back_populates="model")
    metric_points: List["MetricPoint"] = Relationship(back_populates="model")


class Experiment(SQLModel, table=True):
    """Experiment entity for tracking individual ML experiments."""

    __tablename__ = "experiments"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, index=True)
    description: str = Field(default="", max_length=1000)
    status: ExperimentStatus = Field(default=ExperimentStatus.RUNNING)
    project_id: int = Field(foreign_key="projects.id")
    model_id: Optional[int] = Field(foreign_key="models.id")
    dataset_id: Optional[int] = Field(foreign_key="datasets.id")
    clearml_experiment_id: Optional[str] = Field(default=None, max_length=200, index=True)
    clearml_url: Optional[str] = Field(default=None, max_length=500)
    configuration: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    project: Project = Relationship(back_populates="experiments")
    model: Optional[Model] = Relationship(back_populates="experiments")
    dataset: Optional[Dataset] = Relationship(back_populates="experiments")
    metric_points: List["MetricPoint"] = Relationship(back_populates="experiment")


class MetricTarget(SQLModel, table=True):
    """Target thresholds for metrics per project."""

    __tablename__ = "metric_targets"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    metric_name: str = Field(max_length=100, index=True)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    target_value: Decimal = Field(decimal_places=6, max_digits=10)
    comparison_operator: ComparisonOperator = Field(default=ComparisonOperator.GREATER_THAN_EQUAL)
    description: str = Field(default="", max_length=500)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    project: Project = Relationship(back_populates="metric_targets")


class MetricPoint(SQLModel, table=True):
    """Standardized metric data point from ClearML adapters."""

    __tablename__ = "metric_points"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    experiment_id: int = Field(foreign_key="experiments.id")
    model_id: Optional[int] = Field(foreign_key="models.id")
    dataset_id: Optional[int] = Field(foreign_key="datasets.id")
    metric_name: str = Field(max_length=100, index=True)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    value: Decimal = Field(decimal_places=6, max_digits=10)
    timestamp: datetime = Field(index=True)
    iteration: Optional[int] = Field(default=None)
    epoch: Optional[int] = Field(default=None)
    extra_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    experiment: Experiment = Relationship(back_populates="metric_points")
    model: Optional[Model] = Relationship(back_populates="metric_points")
    dataset: Optional[Dataset] = Relationship(back_populates="metric_points")


class CacheEntry(SQLModel, table=True):
    """Cache for improved performance of dashboard queries."""

    __tablename__ = "cache_entries"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    cache_key: str = Field(max_length=500, unique=True, index=True)
    data: Dict[str, Any] = Field(sa_column=Column(JSON))
    expires_at: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectConfig(SQLModel, table=True):
    """Configuration settings for projects and dashboard behavior."""

    __tablename__ = "project_configs"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", unique=True)
    display_name: str = Field(max_length=200)
    tab_order: int = Field(default=0)
    default_chart_type: str = Field(default="line", max_length=50)
    refresh_interval_minutes: int = Field(default=30)
    visible_metrics: List[str] = Field(default=[], sa_column=Column(JSON))
    chart_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    filter_config: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MetricRegistry(SQLModel, table=True):
    """Registry for available metrics and their configuration."""

    __tablename__ = "metric_registry"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    metric_name: str = Field(max_length=100, unique=True, index=True)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    display_name: str = Field(max_length=100)
    description: str = Field(default="", max_length=500)
    unit: str = Field(default="", max_length=20)
    format_string: str = Field(default="{:.4f}", max_length=50)
    higher_is_better: bool = Field(default=True)
    default_target_operator: ComparisonOperator = Field(default=ComparisonOperator.GREATER_THAN_EQUAL)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Non-persistent schemas (for validation, forms, API requests/responses)
class ProjectCreate(SQLModel, table=False):
    """Schema for creating new projects."""

    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=1000)
    clearml_project_name: Optional[str] = Field(default=None, max_length=200)
    adapter_config: Dict[str, Any] = Field(default={})


class ProjectUpdate(SQLModel, table=False):
    """Schema for updating projects."""

    name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    status: Optional[ProjectStatus] = Field(default=None)
    clearml_project_name: Optional[str] = Field(default=None, max_length=200)
    adapter_config: Optional[Dict[str, Any]] = Field(default=None)


class ExperimentCreate(SQLModel, table=False):
    """Schema for creating new experiments."""

    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=1000)
    project_id: int
    model_id: Optional[int] = Field(default=None)
    dataset_id: Optional[int] = Field(default=None)
    clearml_experiment_id: Optional[str] = Field(default=None, max_length=200)
    clearml_url: Optional[str] = Field(default=None, max_length=500)
    configuration: Dict[str, Any] = Field(default={})


class ExperimentUpdate(SQLModel, table=False):
    """Schema for updating experiments."""

    name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    status: Optional[ExperimentStatus] = Field(default=None)
    model_id: Optional[int] = Field(default=None)
    dataset_id: Optional[int] = Field(default=None)
    clearml_url: Optional[str] = Field(default=None, max_length=500)
    configuration: Optional[Dict[str, Any]] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


class ModelCreate(SQLModel, table=False):
    """Schema for creating new models."""

    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=1000)
    architecture: str = Field(default="", max_length=200)
    version: str = Field(default="1.0", max_length=50)
    hyperparameters: Dict[str, Any] = Field(default={})


class DatasetCreate(SQLModel, table=False):
    """Schema for creating new datasets."""

    name: str = Field(max_length=200)
    description: str = Field(default="", max_length=1000)
    version: str = Field(default="1.0", max_length=50)
    project_id: int
    clearml_dataset_id: Optional[str] = Field(default=None, max_length=200)
    extra_data: Dict[str, Any] = Field(default={})


class MetricTargetCreate(SQLModel, table=False):
    """Schema for creating metric targets."""

    project_id: int
    metric_name: str = Field(max_length=100)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    target_value: Decimal
    comparison_operator: ComparisonOperator = Field(default=ComparisonOperator.GREATER_THAN_EQUAL)
    description: str = Field(default="", max_length=500)


class MetricPointCreate(SQLModel, table=False):
    """Schema for creating metric points (from ClearML adapters)."""

    project_id: int
    experiment_id: int
    model_id: Optional[int] = Field(default=None)
    dataset_id: Optional[int] = Field(default=None)
    metric_name: str = Field(max_length=100)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    value: Decimal
    timestamp: datetime
    iteration: Optional[int] = Field(default=None)
    epoch: Optional[int] = Field(default=None)
    extra_data: Dict[str, Any] = Field(default={})


class MetricRegistryCreate(SQLModel, table=False):
    """Schema for registering new metrics."""

    metric_name: str = Field(max_length=100)
    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    display_name: str = Field(max_length=100)
    description: str = Field(default="", max_length=500)
    unit: str = Field(default="", max_length=20)
    format_string: str = Field(default="{:.4f}", max_length=50)
    higher_is_better: bool = Field(default=True)
    default_target_operator: ComparisonOperator = Field(default=ComparisonOperator.GREATER_THAN_EQUAL)


class ProjectConfigCreate(SQLModel, table=False):
    """Schema for creating project configurations."""

    project_id: int
    display_name: str = Field(max_length=200)
    tab_order: int = Field(default=0)
    default_chart_type: str = Field(default="line", max_length=50)
    refresh_interval_minutes: int = Field(default=30)
    visible_metrics: List[str] = Field(default=[])
    chart_config: Dict[str, Any] = Field(default={})
    filter_config: Dict[str, Any] = Field(default={})


class DashboardData(SQLModel, table=False):
    """Schema for dashboard data responses."""

    project_id: int
    project_name: str
    latest_metrics: List[Dict[str, Any]] = Field(default=[])
    metric_trends: List[Dict[str, Any]] = Field(default=[])
    recent_experiments: List[Dict[str, Any]] = Field(default=[])
    comparison_data: List[Dict[str, Any]] = Field(default=[])
    available_filters: Dict[str, List[str]] = Field(default={})
    last_updated: datetime


class MetricComparison(SQLModel, table=False):
    """Schema for metric comparison data."""

    metric_name: str
    datasets: List[str] = Field(default=[])
    models: List[str] = Field(default=[])
    experiments: List[str] = Field(default=[])
    comparison_type: str  # "dataset", "model", "experiment"
    chart_data: List[Dict[str, Any]] = Field(default=[])


class KPICard(SQLModel, table=False):
    """Schema for KPI card display data."""

    metric_name: str
    current_value: Decimal
    target_value: Optional[Decimal] = Field(default=None)
    target_operator: Optional[ComparisonOperator] = Field(default=None)
    is_meeting_target: Optional[bool] = Field(default=None)
    trend_direction: Optional[str] = Field(default=None)  # "up", "down", "stable"
    last_updated: datetime
    unit: str = Field(default="")
    format_string: str = Field(default="{:.4f}")
