"""Unified pipeline API: one router per stage, every stage a background job with a status endpoint."""

from app.pipeline.stages import capture, export, phone, points, splats, system, tracking, walk

routers = [system.router, capture.router, tracking.router, points.router, splats.router, walk.router, export.router, phone.router]
