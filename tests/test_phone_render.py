import io
import json
import zipfile

import numpy as np
import pytest

from app.pipeline.stages import phone


pytestmark = pytest.mark.anyio

NAMES = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]


def _gaussian_ply(path, rows):
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n"
    header += "".join(f"property float {n}\n" for n in NAMES) + "end_header\n"
    path.write_bytes(header.encode() + np.array(rows, "<f4").tobytes())


def test_splat_to_dot_splat_keeps_visible_splats_near_the_cameras(tmp_path):
    grey = [0.0, 0.0, 0.0]
    _gaussian_ply(tmp_path / "in.ply", [
        [0.5, 1.0, -1.0, *grey, 4.0, np.log(0.02), np.log(0.02), np.log(0.02), 2.0, 0.0, 0.0, 0.0],  # kept
        [0.0, 1.0, -1.0, *grey, -6.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # nearly transparent
        [90.0, 1.0, -1.0, *grey, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # floater far from the path
    ])
    info = phone.splat_to_dot_splat(tmp_path / "in.ply", tmp_path / "out.splat", np.array([[0, 1.4, 0], [1, 1.4, 0]]), 10)
    assert info["splats_in"] == 3 and info["splats_out"] == 1 and info["bytes"] == 32
    record = np.frombuffer((tmp_path / "out.splat").read_bytes(),
                           dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])[0]
    assert np.allclose(record["p"], [0.5, 1.0, -1.0])
    assert np.allclose(record["s"], 0.02, atol=1e-6)
    assert list(record["c"][:3]) == [128, 128, 128] and record["c"][3] == 250  # sigmoid(4) * 255 = 250.4
    assert list(record["r"]) == [255, 128, 128, 128]  # normalised (1, 0, 0, 0)


def _bundle(mode="ar") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("meta.json", json.dumps({"format": "namma-sensor-bundle", "mode": mode}))
        archive.writestr("padding.bin", bytes(range(256)) * 4)
    return buffer.getvalue()


async def test_chunked_upload_starts_a_render(client, monkeypatch):
    submitted = {}

    class FakeJob:
        id = "job-1"

    def fake_submit(db, project_id, kind, params):
        submitted.update(project_id=project_id, kind=kind, params=params)
        return FakeJob()

    monkeypatch.setattr(phone, "submit_task", fake_submit)
    monkeypatch.setattr(phone, "CHUNK_BYTES", 400)
    data = _bundle()
    start = (await client.post("/api/v1/remote/uploads", json={"filename": "room.zip", "size": len(data)})).json()
    assert start["chunks"] == -(-len(data) // 400)
    assert (await client.put(f"/api/v1/remote/uploads/{start['upload_id']}/chunks/0", content=b"short")).status_code == 400
    for index in range(start["chunks"] - 1):  # leave the last one out
        response = await client.put(f"/api/v1/remote/uploads/{start['upload_id']}/chunks/{index}", content=data[index * 400:(index + 1) * 400])
        assert response.status_code == 200
    missing = await client.post(f"/api/v1/remote/uploads/{start['upload_id']}/finish", json={"quality": "fast"})
    assert missing.status_code == 409 and missing.json()["detail"]["missing"] == [start["chunks"] - 1]
    last = start["chunks"] - 1
    await client.put(f"/api/v1/remote/uploads/{start['upload_id']}/chunks/{last}", content=data[last * 400:])
    status = (await client.get(f"/api/v1/remote/uploads/{start['upload_id']}")).json()
    assert status["received"] == list(range(start["chunks"]))
    finish = (await client.post(f"/api/v1/remote/uploads/{start['upload_id']}/finish", json={"quality": "fast"})).json()
    assert finish["render_id"] == "job-1" and finish["kind"] == "ar"
    assert submitted["kind"] == "phone.render" and submitted["params"]["quality"] == "fast"
    again = (await client.post(f"/api/v1/remote/uploads/{start['upload_id']}/finish", json={"quality": "fast"})).json()
    assert again["render_id"] == "job-1"  # a retried finish does not start a second render


async def test_rejects_other_files(client):
    response = await client.post("/api/v1/remote/uploads", json={"filename": "notes.txt", "size": 10})
    assert response.status_code == 400


async def test_pairing_key(client, monkeypatch):
    monkeypatch.setenv("NAMMA_REMOTE_KEY", "secret")
    assert (await client.get("/api/v1/remote/hello")).status_code == 401
    assert (await client.get("/api/v1/remote/hello", headers={"X-Namma-Key": "wrong"})).status_code == 401
    assert (await client.get("/api/v1/remote/hello", headers={"X-Namma-Key": "secret"})).status_code == 200
    assert (await client.get("/api/v1/remote/hello?key=secret")).status_code == 200


def test_render_plan_uses_ar_poses_for_ar_bundles():
    steps = phone.render_plan("ar", phone.QUALITY["fast"], None)
    assert [s["task"] for s in steps] == ["capture.bundle", "tracking.ar_poses", "phone.dataset", "points.monocular", "splats.train"]
    assert steps[-1]["optional"] and "no trainer" in steps[-1]["label"]
    video = phone.render_plan("video", phone.QUALITY["fast"], "gsplat")
    assert [s["task"] for s in video] == ["capture.frames", "tracking.sfm", "points.monocular", "splats.train"]
