"""
Configuration management: YAML defaults, file overrides, CLI overrides.
"""

import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(Exception):
    """Raised when a configuration file or override is invalid."""


@dataclass
class ColumnTypeConfig:
    """Per-column type overrides."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    ignore: List[str] = field(default_factory=list)


@dataclass
class DataQualityConfig:
    """Data quality analysis settings."""

    missing_threshold: float = 0.95  # Flag columns with >95% missing
    cardinality_threshold: int = 100  # Flag >100 unique categories
    duplicate_threshold: float = 0.1  # Flag if >10% duplicates
    constant_threshold: float = 0.99  # Flag if >99% same value


@dataclass
class OutlierConfig:
    """Outlier detection settings."""

    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    isolation_forest_enabled: bool = True
    # None: the cut comes from each dataset's anomaly scores. A number flags that fixed share of rows.
    isolation_forest_contamination: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Visualization settings."""

    max_histograms: int = 20
    max_boxplots: int = 20
    max_correlation_heatmap_size: int = 30
    max_scatter_pairs: int = 10
    correlation_threshold: float = 0.05  # Min abs(corr) to show


@dataclass
class TargetConfig:
    """Target variable analysis."""

    target_column: Optional[str] = None
    target_type: Optional[str] = None  # 'classification', 'regression', auto-detect
    class_imbalance_threshold: float = 0.8


@dataclass
class Config:
    """Complete EDA pipeline configuration."""

    # Data input
    input_file: Optional[str] = None
    input_folder: Optional[str] = None
    file_format: Optional[str] = None  # csv, xlsx, parquet, json; auto-detect if None
    delimiter: Optional[str] = None  # Auto-detect if None
    encoding: Optional[str] = None  # Auto-detect if None
    decimal: Optional[str] = None  # '.' or ','; auto-detect if None
    excel_sheet: Optional[str] = None  # Sheet name or index; first sheet if None
    sample_size: Optional[int] = None  # Random sample for large datasets
    batch_pattern: Optional[str] = None  # Glob pattern for analyze-batch; all supported extensions if None

    # Analysis configuration
    column_types: ColumnTypeConfig = field(default_factory=ColumnTypeConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    visualizations: VisualizationConfig = field(default_factory=VisualizationConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Output
    output_dir: str = "reports"
    strict_mode: bool = False  # Fail (raise) on first dataset/step error if True
    verbose: bool = False  # DEBUG-level console logging if True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Load from dictionary (with nested dataclass handling).

        Raises:
            ConfigError: an unknown key was supplied (top-level or nested), or
                a value fails validation (see ``validate``).
        """
        data = dict(data)  # avoid mutating the caller's dict

        if "language" in data:
            warnings.warn(
                "The 'language' configuration key is obsolete and no longer has any effect; it is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.pop("language")

        nested_dataclasses = {
            "column_types": ColumnTypeConfig,
            "data_quality": DataQualityConfig,
            "outliers": OutlierConfig,
            "visualizations": VisualizationConfig,
            "target": TargetConfig,
        }
        for key, dataclass_type in nested_dataclasses.items():
            if key in data and isinstance(data[key], dict):
                data[key] = _instantiate_section(dataclass_type, data[key], key)

        _reject_unknown_keys(cls, data, "configuration")

        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate value ranges, raising ConfigError with a Spanish message on failure."""
        _require_ratio("data_quality.missing_threshold", self.data_quality.missing_threshold)
        _require_ratio("data_quality.duplicate_threshold", self.data_quality.duplicate_threshold)
        _require_ratio("data_quality.constant_threshold", self.data_quality.constant_threshold)
        _require_positive_int("data_quality.cardinality_threshold", self.data_quality.cardinality_threshold)

        _require_positive("outliers.iqr_multiplier", self.outliers.iqr_multiplier)
        _require_positive("outliers.z_score_threshold", self.outliers.z_score_threshold)
        if self.outliers.isolation_forest_contamination is not None:
            _require_ratio("outliers.isolation_forest_contamination", self.outliers.isolation_forest_contamination)

        _require_positive_int("visualizations.max_histograms", self.visualizations.max_histograms)
        _require_positive_int("visualizations.max_boxplots", self.visualizations.max_boxplots)
        _require_positive_int(
            "visualizations.max_correlation_heatmap_size", self.visualizations.max_correlation_heatmap_size
        )
        _require_positive_int("visualizations.max_scatter_pairs", self.visualizations.max_scatter_pairs)
        _require_ratio(
            "visualizations.correlation_threshold", self.visualizations.correlation_threshold, allow_zero=True
        )

        _require_ratio("target.class_imbalance_threshold", self.target.class_imbalance_threshold)
        if self.target.target_type is not None and self.target.target_type not in ("classification", "regression"):
            raise ConfigError(
                f"Invalid value for 'target.target_type': {self.target.target_type!r}. "
                "Valid values: 'classification', 'regression' or null (auto-detect)."
            )

        if self.sample_size is not None:
            _require_positive_int("sample_size", self.sample_size)

        if self.decimal is not None and self.decimal not in (".", ","):
            raise ConfigError(f"Invalid value for 'decimal': {self.decimal!r}. Use '.', ',' or null (auto-detect).")


def _valid_field_names(dataclass_or_type: Any) -> set:
    return {f.name for f in fields(dataclass_or_type)}


def _reject_unknown_keys(dataclass_type: Any, data: Dict[str, Any], section: str) -> None:
    """Raise ConfigError if `data` has keys not defined on `dataclass_type`."""
    valid_keys = _valid_field_names(dataclass_type)
    unknown = sorted(set(data) - valid_keys)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in the '{section}' section: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(valid_keys))}."
        )


def _instantiate_section(dataclass_type: Any, data: Dict[str, Any], section: str) -> Any:
    """Build a nested config dataclass, raising a clear error on unknown keys."""
    _reject_unknown_keys(dataclass_type, data, section)
    return dataclass_type(**data)


def _require_ratio(name: str, value: float, allow_zero: bool = False) -> None:
    """Validate that `value` is in (0, 1] (or [0, 1] if allow_zero)."""
    lower_ok = value >= 0 if allow_zero else value > 0
    if not (lower_ok and value <= 1):
        bounds = "[0, 1]" if allow_zero else "(0, 1]"
        raise ConfigError(f"'{name}' must be within {bounds}; got {value!r}.")


def _require_positive(name: str, value: float) -> None:
    if not value > 0:
        raise ConfigError(f"'{name}' must be a positive number; got {value!r}.")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer; got {value!r}.")


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_NESTED_SECTION_KEYS = ("column_types", "data_quality", "outliers", "visualizations", "target")


def _merge_section(current: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge a nested config section, keeping keys not present in `updates`."""
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(updates)
    return merged


def merge_configs(defaults: Config, file_config: Dict[str, Any], cli_overrides: Dict[str, Any]) -> Config:
    """
    Merge configuration layers: defaults < file < CLI.

    Nested sections (``column_types``, ``data_quality``, ``outliers``,
    ``visualizations``, ``target``) are deep-merged key by key, so e.g. a
    CLI-provided ``target.target_column`` does not wipe out a YAML-provided
    ``target.class_imbalance_threshold``. Scalar CLI overrides are applied
    only when not None, so options the user did not pass never clobber a
    YAML value.
    """
    config_dict = defaults.to_dict()

    for key, value in file_config.items():
        if key in _NESTED_SECTION_KEYS and isinstance(value, dict):
            config_dict[key] = _merge_section(config_dict.get(key), value)
        else:
            config_dict[key] = value

    for key, value in cli_overrides.items():
        if value is None:
            continue
        if key in _NESTED_SECTION_KEYS and isinstance(value, dict):
            config_dict[key] = _merge_section(config_dict.get(key), value)
        else:
            config_dict[key] = value

    return Config.from_dict(config_dict)


def create_default_config_file(output_path: Path) -> None:
    """Create a default config YAML file."""
    default_config = {
        "file_format": None,
        "delimiter": None,
        "encoding": None,
        "decimal": None,
        "excel_sheet": None,
        "sample_size": None,
        "batch_pattern": None,
        "column_types": {"numeric": [], "categorical": [], "datetime": [], "text": [], "ignore": []},
        "data_quality": {
            "missing_threshold": 0.95,
            "cardinality_threshold": 100,
            "duplicate_threshold": 0.1,
            "constant_threshold": 0.99,
        },
        "outliers": {
            "iqr_multiplier": 1.5,
            "z_score_threshold": 3.0,
            "isolation_forest_enabled": True,
            "isolation_forest_contamination": None,
        },
        "visualizations": {
            "max_histograms": 20,
            "max_boxplots": 20,
            "max_correlation_heatmap_size": 30,
            "max_scatter_pairs": 10,
            "correlation_threshold": 0.05,
        },
        "target": {"target_column": None, "target_type": None, "class_imbalance_threshold": 0.8},
        "output_dir": "reports",
        "strict_mode": False,
        "verbose": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
