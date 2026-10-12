from pathlib import Path

import pytest


pytestmark = pytest.mark.anyio


async def test_health_reports_runtime_tools(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert Path(payload["ffmpeg"]).is_file()


async def test_upload_creates_project_and_persists_video(client):
    response = await client.post(
        "/api/v1/projects",
        data={"name": "Demo room"},
        files={"video": ("room.mp4", b"not-empty", "video/mp4")},
    )
    assert response.status_code == 201
    project = response.json()
    assert project["name"] == "Demo room"
    assert project["status"] == "imported"
    assert project["source_size_bytes"] == 9
    assert len(project["source_sha256"]) == 64
    fetched = await client.get(f"/api/v1/projects/{project['id']}")
    assert fetched.status_code == 200


async def test_upload_rejects_unknown_extension(client):
    response = await client.post(
        "/api/v1/projects",
        data={"name": "Bad input"},
        files={"video": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


async def test_upload_rejects_blank_project_name(client):
    response = await client.post(
        "/api/v1/projects",
        data={"name": "   "},
        files={"video": ("room.mp4", b"video", "video/mp4")},
    )
    assert response.status_code == 400
