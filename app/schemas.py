from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    source_filename: str
    source_size_bytes: int
    source_sha256: str
    frame_count: int
    registered_image_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    kind: str
    status: str
    stage: str
    progress: float
    log_path: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class HealthRead(BaseModel):
    status: str
    ffmpeg: str
    colmap: str | None


class ScaleCalibrationRequest(BaseModel):
    known_distance_meters: float
    observed_distance_units: float
    reference: str = "manual room measurement"


class NavigationApprovalRequest(BaseModel):
    reviewer: str
    notes: str = ""
