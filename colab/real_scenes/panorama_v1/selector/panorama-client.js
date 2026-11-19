import {
  oauthLoginUrl,
  oauthHandleRedirectIfPresent,
  whoAmI,
  uploadFilesWithProgress,
  downloadFile
} from "https://esm.sh/@huggingface/hub@2.17.10?bundle";
import { Client as GradioClient } from "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.2/dist/index.min.js";

const INBOX = "anasrazy/namma-panorama-inbox";
const OUTPUT = "anasrazy/namma-panorama-out";
const GRADIO_SPACE = "BilalSardar/Google-Street-View-360-panorama-Image-Downloader";
const GRADIO_ORIGIN = "https://bilalsardar-google-street-view-360-panorama-imag-ceccea6.hf.space";
const TOUR_ORIGIN = "https://anasrazy-namma-tour.static.hf.space";
const HUB_REPO = { type: "dataset", name: INBOX };
const OUTPUT_REPO = { type: "dataset", name: OUTPUT };
const $ = (id) => document.getElementById(id);
let accessToken = null;
let activeJobId = null;
let activeGradioClient = null;
let oauthReady = false;

function renderStatus(message, percent = null, detail = "") {
  const text = $("panoramaJobText");
  const bar = $("panoramaJobBar");
  const count = $("panoramaJobCount");
  const wrap = $("panoramaJobProgress");
  if (!text || !bar || !count || !wrap) return;
  wrap.classList.remove("hidden");
  text.textContent = message;
  if (percent == null) {
    bar.style.width = "35%";
    bar.classList.add("is-indeterminate");
  } else {
    bar.classList.remove("is-indeterminate");
    bar.style.width = Math.max(0, Math.min(100, percent)) + "%";
  }
  count.textContent = detail;
}

function failStatus(error) {
  const message = error && error.message ? error.message : String(error);
  renderStatus("Could not submit the panorama job: " + message, 0);
  console.error("[NAMMA panorama]", message);
}

function makeStreetViewUrl(panoId) {
  return "https://www.google.com/maps/@0,0,3a,75y,0h,90t/data=!3m7!1e1!3m5!1s" +
    encodeURIComponent(panoId) + "!2e0";
}

async function getDownloadedPanoramaBlob(result) {
  const data = result && Array.isArray(result.data) ? result.data[0] : null;
  if (!data) throw new Error("The Street View downloader returned no image.");
  const candidates = [];
  if (typeof data === "string") candidates.push(data);
  if (data && typeof data === "object") {
    if (typeof data.url === "string") candidates.push(data.url);
    if (typeof data.path === "string") candidates.push(data.path);
    if (typeof data.name === "string") candidates.push(data.name);
  }
  if (data && typeof data.blob === "function") {
    const blob = await data.blob();
    if (blob && blob.size) return blob;
  }

  let lastError = null;
  for (const candidate of candidates) {
    const url = /^https?:\/\//i.test(candidate)
      ? candidate
      : GRADIO_ORIGIN + "/gradio_api/file=" + encodeURIComponent(candidate.replace(/^\/+/, ""));
    try {
      const response = await fetch(url, { credentials: "omit" });
      if (!response.ok) throw new Error("Image download HTTP " + response.status);
      const blob = await response.blob();
      if (blob.size > 0) return blob;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error("The downloader finished but its image file could not be fetched." +
    (lastError ? " " + lastError.message : ""));
}

async function validateEquirectangular(blob, panoId) {
  if (!blob || !blob.size) throw new Error("The downloader returned an empty panorama for " + panoId + ".");
  if (typeof createImageBitmap !== "function") return blob;
  const bitmap = await createImageBitmap(blob);
  try {
    const ratio = bitmap.width / bitmap.height;
    if (bitmap.width < 2048 || bitmap.height < 1024 || ratio < 1.75 || ratio > 2.25) {
      throw new Error(
        "Panorama " + panoId + " is " + bitmap.width + "×" + bitmap.height +
        ", not a full-resolution 2:1 equirectangular image. Increase downloader zoom/grid and retry."
      );
    }
  } finally {
    bitmap.close();
  }
  return blob;
}

async function runGradioDownload(pano, index, total, settings) {
  if (!activeGradioClient) {
    renderStatus("Connecting to the Street View image service…", null);
    activeGradioClient = await GradioClient.connect(GRADIO_SPACE, { events: ["status"] });
  }
  const result = await activeGradioClient.predict("/process_street_view", [
    makeStreetViewUrl(pano.panoId),
    settings.zoom,
    settings.columns,
    settings.rows
  ]);
  const blob = await getDownloadedPanoramaBlob(result);
  await validateEquirectangular(blob, pano.panoId);
  renderStatus(
    "Downloaded panorama " + index + " of " + total,
    15 + 35 * index / total,
    Math.round(blob.size / 1048576) + " MB · " + pano.panoId.slice(0, 12)
  );
  return blob;
}

async function uploadOne(path, blob, title, token) {
  for await (const event of uploadFilesWithProgress({
    repo: HUB_REPO,
    accessToken: token,
    files: [{ path, content: blob }],
    commitTitle: title,
    useWebWorkers: true
  })) {
    if (event && event.event === "fileProgress") {
      const pct = Math.round(event.progress || 0);
      renderStatus("Uploading panorama data to your private inbox…", null, pct + "%");
    }
  }
}

async function uploadJson(path, value, title, token) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  await uploadOne(path, blob, title, token);
}

function makeJobId() {
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
  const nonce = Math.random().toString(36).slice(2, 8);
  return "street-" + stamp + "-" + nonce;
}

function setTourLink(url) {
  const link = $("panoramaJobLink");
  if (!link) return;
  link.href = url;
  link.textContent = "Open 3D tour ↗";
  link.classList.remove("hidden");
  link.target = "_blank";
  link.rel = "noopener noreferrer";
}

function isMissingStatusFile(error) {
  const status = Number(
    error && (error.status || error.statusCode ||
      error.response && error.response.status ||
      error.cause && error.cause.status)
  );
  const message = String(error && error.message || error || "");
  return status === 404 || /\\b404\\b|EntryNotFoundError|RepositoryNotFoundError|not found/i.test(message);
}

async function pollJob(jobId, token) {
  const statusPath = jobId + "/status.json";
  const started = Date.now();
  while (Date.now() - started < 6 * 60 * 60 * 1000) {
    await new Promise((resolve) => window.setTimeout(resolve, 20000));
    let blob = null;
    try {
      blob = await downloadFile({
        repo: OUTPUT_REPO,
        path: statusPath,
        accessToken: token
      });
    } catch (error) {
      if (!isMissingStatusFile(error)) throw error;
    }
    // The status file is created by the scheduled worker, which may not have
    // processed the job yet. A 404 during this period means "queued", not failure.
    if (!blob) {
      renderStatus("Job uploaded. Waiting for the cloud GPU queue to pick it up…", 50);
      continue;
    }
    let status;
    try {
      status = JSON.parse(await blob.text());
    } catch {
      renderStatus("Job uploaded. Waiting for a valid cloud status update…", 50);
      continue;
    }
    const phase = String(status.status || "queued");
    const percent = ({
      staging: 52, queued: 58, training: 78, publishing: 94, complete: 100, failed: 0
    })[phase] ?? 60;
    renderStatus(
      "Panorama reconstruction: " + phase,
      percent,
      status.registered_views ? status.registered_views + " registered views" : ""
    );
    if (phase === "complete" && status.tour_url) {
      setTourLink(status.tour_url);
      return;
    }
    if (phase === "failed") {
      throw new Error(status.error || "The cloud worker marked this job as failed.");
    }
  }
  renderStatus("The job is still in the cloud queue. You can return to this selector and check again later.", 60);
}

async function authenticateAndResume() {
  const oauth = await oauthHandleRedirectIfPresent();
  if (!oauth || !oauth.accessToken) return;
  const profile = await whoAmI({ accessToken: oauth.accessToken });
  const username = profile && (profile.name || profile.username || profile.user && profile.user.name);
  if (String(username || "").toLowerCase() !== "anasrazy") {
    throw new Error("Sign in with the Hugging Face account that owns NAMMA's private panorama inbox.");
  }
  accessToken = oauth.accessToken;
  oauthReady = true;
  const pending = sessionStorage.getItem("namma-panorama-pending");
  if (pending) {
    sessionStorage.removeItem("namma-panorama-pending");
    const parsed = JSON.parse(pending);
    if (Array.isArray(parsed.panoramaSelection) && parsed.panoramaSelection.length >= 8) {
      await submitPanoramaJob(parsed.panoramaSelection, parsed.settings);
    }
  }
}

async function submitPanoramaJob(selection, settings = {}) {
  if (!Array.isArray(selection) || selection.length < 8 || selection.length > 200) {
    throw new Error("Select between 8 and 200 panorama positions.");
  }
  if (!oauthReady || !accessToken) {
    sessionStorage.setItem("namma-panorama-pending", JSON.stringify({
      panoramaSelection: selection,
      settings
    }));
    const loginUrl = await oauthLoginUrl();
    window.location.assign(loginUrl);
    return;
  }

  const profile = await whoAmI({ accessToken });
  const username = profile && (profile.name || profile.username || profile.user && profile.user.name);
  if (String(username || "").toLowerCase() !== "anasrazy") {
    accessToken = null;
    oauthReady = false;
    throw new Error("Only the NAMMA Hugging Face owner account can submit to this private queue.");
  }

  const normalizedSelection = selection.map((item, index) => {
    if (!item || !/^[A-Za-z0-9_-]+$/.test(String(item.panoId || ""))) {
      throw new Error("A selected panorama ID contains unsupported characters.");
    }
    const lat = item.lat == null || item.lat === "" ? null : Number(item.lat);
    const lonValue = item.lng ?? item.lon;
    const lon = lonValue == null || lonValue === "" ? null : Number(lonValue);
    if ((lat === null) !== (lon === null)) {
      throw new Error("A panorama must have both latitude and longitude, or neither.");
    }
    if (lat !== null && (!Number.isFinite(lat) || lat < -90 || lat > 90 ||
        !Number.isFinite(lon) || lon < -180 || lon > 180)) {
      throw new Error("A panorama contains an invalid GPS coordinate.");
    }
    const sequence = Number.isFinite(Number(item.sequence)) ? Number(item.sequence) : index;
    return { ...item, panoId: String(item.panoId), lat, lng: lon, sequence };
  });
  const selected = normalizedSelection.sort((a, b) => a.sequence - b.sequence);
  const jobId = makeJobId();
  activeJobId = jobId;
  const zoom = Number(settings.zoom ?? 4);
  const columns = Number(settings.columns ?? 44);
  const rows = Number(settings.rows ?? 22);
  if (!Number.isInteger(zoom) || zoom < 0 || zoom > 5 ||
      !Number.isInteger(columns) || columns < 1 || columns > 128 ||
      !Number.isInteger(rows) || rows < 1 || rows > 64 || columns * rows > 1000) {
    throw new Error("The panorama download grid must be valid and no larger than 1,000 tiles.");
  }

  const jobRoot = jobId + "/";
  const manifest = {
    schema_version: 1,
    scene_name: "NAMMA Street View route",
    panoramas: selected.map((item, index) => ({
      pano_id: item.panoId,
      image: "panoramas/" + String(index).padStart(4, "0") + "-" + item.panoId + ".jpg",
      sequence: index,
      lat: item.lat,
      lon: item.lng
    }))
  };
  const remotePaths = [];
  renderStatus("Preparing " + selected.length + " panoramas for upload…", 15);

  for (let i = 0; i < selected.length; i++) {
    const pano = selected[i];
    if (!pano || !/^[A-Za-z0-9_-]+$/.test(pano.panoId)) {
      throw new Error("A selected panorama ID contains unsupported characters.");
    }
    const blob = await runGradioDownload(pano, i + 1, selected.length, {
      zoom, columns, rows
    });
    const filename = "panoramas/" + String(i).padStart(4, "0") + "-" + pano.panoId + ".jpg";
    const remotePath = jobRoot + filename;
    renderStatus(
      "Uploading panorama " + (i + 1) + " of " + selected.length + "…",
      50 + 35 * i / selected.length,
      Math.round(blob.size / 1048576) + " MB"
    );
    await uploadOne(remotePath, blob, "NAMMA panorama " + jobId + " image " + (i + 1), accessToken);
    remotePaths.push(remotePath);
  }

  renderStatus("Uploading route manifest…", 87);
  await uploadJson(jobRoot + "manifest.json", manifest, "NAMMA panorama " + jobId + " manifest", accessToken);

  const job = {
    job_id: jobId,
    pipeline: "panorama_v1",
    bundle_mode: "loose",
    hf_dataset: INBOX,
    manifest_path: jobRoot + "manifest.json",
    panorama_paths: remotePaths,
    output_dataset: OUTPUT,
    tour_space: "anasrazy/namma-tour",
    parameters: {
      steps: 30000,
      max_splats: 1500000,
      max_panoramas: selected.length,
      image_size: 1280,
      view_count: 12,
      hfov_deg: 60.0,
      sequence_window: 18,
      retrieval_count: 10
    }
  };
  renderStatus("Publishing the job request to the online queue…", 92);
  await uploadJson(jobRoot + "job.json", job, "NAMMA panorama " + jobId + " READY", accessToken);
  sessionStorage.setItem("namma-panorama-last-job", jobId);
  renderStatus(
    "Job submitted. The online cloud worker will stage the panoramas, reconstruct the route and publish the 3D tour.",
    100,
    selected.length + " panoramas · 1280px training · 30K steps"
  );
  pollJob(jobId, accessToken).catch(failStatus);
}

window.NAMMA_START_PANORAMA = (selection, settings) =>
  submitPanoramaJob(selection, settings).catch((error) => {
    failStatus(error);
    throw error;
  });

try {
  await authenticateAndResume();
} catch (error) {
  failStatus(error);
}
