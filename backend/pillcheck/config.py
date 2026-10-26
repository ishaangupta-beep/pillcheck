"""Runtime configuration, read once from environment variables.

Every threshold the vision pipeline and the agent rely on lives here, so the
evaluation harness can sweep them and the technical report can cite them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BACKEND_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _envf(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _envi(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class VisionConfig:
    models_dir: Path = Path(_env("PILLCHECK_MODELS_DIR", str(BACKEND_ROOT / "models")))
    aliked_model: str = "aliked-n16rot-top1k-640.onnx"
    lightglue_model: str = "aliked_lightglue.onnx"
    aliked_input: int = _envi("PILLCHECK_ALIKED_INPUT", 640)
    # Working resolution: long side of the image fed to the pipeline.
    work_long_side: int = _envi("PILLCHECK_WORK_LONG_SIDE", 1280)

    # Capture-quality gates.
    min_long_side: int = 480
    blur_min_laplacian_var: float = _envf("PILLCHECK_BLUR_MIN", 12.0)
    dark_max_mean: float = 70.0  # applied to the 90th-percentile brightness (the lit subject)
    bright_min_mean: float = 235.0
    glare_pixel_value: int = 250
    glare_max_fraction: float = 0.12

    # Registration acceptance.
    min_inliers: int = _envi("PILLCHECK_MIN_INLIERS", 40)
    min_inlier_ratio: float = 0.25
    max_reproj_rmse_px: float = 4.0
    magsac_threshold_px: float = 3.0
    # Identity: after alignment, the 10th-percentile tile gradient correlation
    # over the printed foil (see verify_appearance). Right strip >= ~0.67,
    # different strip (even same layout, different name) <= ~0.56.
    identity_min_inliers_geometry: int = _envi("PILLCHECK_ID_MIN_INLIERS", 120)
    identity_min_appearance: float = _envf("PILLCHECK_IDENTITY_MIN_APPEARANCE", 0.62)

    # Pocket state decision on the normalised change score (see classify.py).
    colour_unit: float = _envf("PILLCHECK_COLOUR_UNIT", 10.0)  # mean Lab distance
    structure_unit: float = _envf("PILLCHECK_STRUCTURE_UNIT", 0.10)  # 1 - SSIM
    pocket_compare_px: float = _envf("PILLCHECK_POCKET_PX", 64.0)  # pockets are compared at about this diameter
    popped_threshold: float = _envf("PILLCHECK_POPPED_T", 2.4)
    intact_threshold: float = _envf("PILLCHECK_INTACT_T", 2.2)
    pocket_glare_max_fraction: float = 0.35


@dataclass(frozen=True)
class AgentConfig:
    # Bedrock model for the planning step. Empty -> deterministic planner only.
    bedrock_model_id: str = _env("PILLCHECK_BEDROCK_MODEL_ID", "")
    aws_region: str = _env("AWS_REGION", "us-east-1")
    max_steps: int = 6
    # Dose timing windows, in minutes relative to the scheduled time.
    early_window_min: int = _envi("PILLCHECK_EARLY_WINDOW_MIN", 120)
    reminder_after_min: int = _envi("PILLCHECK_REMINDER_AFTER_MIN", 60)
    caregiver_after_min: int = _envi("PILLCHECK_CAREGIVER_AFTER_MIN", 180)
    # A dose more than this late is missed for good: a pill taken later is not
    # credited to it (that would hide a possible double dose with the next one).
    max_late_min: int = _envi("PILLCHECK_MAX_LATE_MIN", 360)
    expiry_soon_days: int = 30
    refill_days_threshold: int = _envi("PILLCHECK_REFILL_DAYS", 3)


@dataclass(frozen=True)
class StorageConfig:
    backend: str = _env("PILLCHECK_STORAGE", "local")  # local | aws
    data_dir: Path = Path(_env("PILLCHECK_DATA_DIR", str(BACKEND_ROOT / "data")))
    ddb_table: str = _env("PILLCHECK_DDB_TABLE", "pillcheck")
    s3_bucket: str = _env("PILLCHECK_S3_BUCKET", "")
    sns_topic_arn: str = _env("PILLCHECK_SNS_TOPIC_ARN", "")


@dataclass(frozen=True)
class LimitsConfig:
    """Spending protection for a public URL. Counters live in the store."""
    max_upload_mb: float = _envf("PILLCHECK_MAX_UPLOAD_MB", 15.0)
    checks_per_ip_hour: int = _envi("PILLCHECK_RL_CHECKS_PER_HOUR", 40)
    references_per_ip_hour: int = _envi("PILLCHECK_RL_REFS_PER_HOUR", 10)
    demo_sessions_per_ip_hour: int = _envi("PILLCHECK_RL_SESSIONS_PER_HOUR", 5)
    checks_per_day_global: int = _envi("PILLCHECK_GLOBAL_CHECKS_PER_DAY", 1500)
    bedrock_runs_per_day: int = _envi("PILLCHECK_BEDROCK_DAILY_CAP", 300)


@dataclass(frozen=True)
class Settings:
    vision: VisionConfig = field(default_factory=VisionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    api_token: str = _env("PILLCHECK_API_TOKEN", "")
    default_timezone: str = _env("PILLCHECK_TZ", "Asia/Kolkata")


settings = Settings()
