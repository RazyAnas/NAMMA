from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "NAMMA_progress_summary.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE9"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(18 * mm, 9 * mm, "NAMMA | Student technology demonstration")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=25, leading=30, textColor=colors.HexColor("#102A43"),
    alignment=TA_CENTER, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=16, textColor=colors.HexColor("#52606D"),
    alignment=TA_CENTER, spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=16, leading=20, textColor=colors.HexColor("#0B7285"),
    spaceBefore=10, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=11.5, leading=15, textColor=colors.HexColor("#102A43"),
    spaceBefore=7, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=14, textColor=colors.HexColor("#243B53"),
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8, leading=11, textColor=colors.HexColor("#52606D"),
))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10, leading=14, textColor=colors.HexColor("#102A43"),
    backColor=colors.HexColor("#E6FFFA"), borderColor=colors.HexColor("#81E6D9"),
    borderWidth=0.6, borderPadding=8, spaceBefore=5, spaceAfter=10,
))


def P(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def bullet(text):
    return P(f"&bull;&nbsp;&nbsp;{text}", "BodyCustom")


story = []
story.append(Spacer(1, 13 * mm))
story.append(P("NAMMA", "TitleCustom"))
story.append(P("Video to photorealistic 3D with obstacle-aware laptop navigation", "Subtitle"))
story.append(P("Progress summary | 9 October 2026", "Small"))
story.append(Spacer(1, 7 * mm))
story.append(P(
    "This report records the architecture, technology decisions, project files and local environment work completed so far. "
    "The project is scoped as a non-commercial student technology demonstration: a prerecorded monocular RGB video becomes a photorealistic 3D scene that can be explored on a laptop without moving through reconstructed obstacles.",
    "Callout"
))

story.append(P("Current status", "H1Custom"))
status_data = [
    [P("Area", "Small"), P("Status", "Small"), P("Result", "Small")],
    [P("Architecture", "Small"), P("Complete", "Small"), P("Defined the visual, geometry, navigation, tracking and viewer layers.", "Small")],
    [P("Repository setup", "Small"), P("Complete", "Small"), P("Created architecture, requirements, environment and verification files.", "Small")],
    [P("Python environment", "Small"), P("Complete", "Small"), P("Python 3.11 virtual environment created; all requirements installed and validated with pip check plus the environment verifier.", "Small")],
    [P("GPU check", "Small"), P("Complete", "Small"), P("NVIDIA RTX 5060-series laptop GPU, approximately 8 GB VRAM detected.", "Small")],
    [P("Native reconstruction tools", "Small"), P("Not installed", "Small"), P("COLMAP, FFmpeg, CMake, WSL and CUDA compiler are not currently on PATH.", "Small")],
    [P("MODP integration", "Small"), P("Not started", "Small"), P("Requires Linux/CUDA-compatible research setup and a tracker-to-MODP adapter.", "Small")],
    [P("Viewer", "Small"), P("Not started", "Small"), P("PlayCanvas/WebGPU laptop viewer will consume packaged splat, collision and NavMesh assets.", "Small")],
]
tbl = Table(status_data, colWidths=[34*mm, 29*mm, 111*mm], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B7285")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E0")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
]))
story.append(tbl)

story.append(P("Architecture agreed", "H1Custom"))
story.append(P("The system keeps separate representations for appearance, physical geometry and navigation:", "BodyCustom"))
arch_data = [
    [P("Layer", "Small"), P("Technology or method", "Small"), P("Purpose", "Small")],
    [P("Camera and sparse geometry", "Small"), P("COLMAP first; ORB-SLAM3 later", "Small"), P("Camera poses, intrinsics and sparse 3D points", "Small")],
    [P("Photorealistic map", "Small"), P("MODP / online Gaussian mapping", "Small"), P("Detailed view synthesis from monocular RGB", "Small")],
    [P("Physical geometry", "Small"), P("Dense multi-view stereo, TSDF or surface mesh", "Small"), P("Walls, floors and obstacle surfaces", "Small")],
    [P("Navigation", "Small"), P("Recast/Detour plus capsule controller", "Small"), P("Walkable polygons, pathfinding and collision-safe movement", "Small")],
    [P("Laptop viewer", "Small"), P("PlayCanvas/SuperSplat with WebGPU", "Small"), P("Render splats and support desktop navigation", "Small")],
    [P("Persistence", "Small"), P("SQLite plus filesystem artifacts", "Small"), P("Jobs, scene versions, paths, hashes and metrics", "Small")],
]
arch_tbl = Table(arch_data, colWidths=[39*mm, 57*mm, 78*mm], repeatRows=1)
arch_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#102A43")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E0")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
]))
story.append(arch_tbl)

story.append(PageBreak())
story.append(P("What has been designed", "H1Custom"))
story.append(P("The agreed data flow is:", "BodyCustom"))
story.append(P("RGB video &rarr; keyframe selection and calibration &rarr; camera tracking and sparse points &rarr; MODP Gaussian reconstruction &rarr; dense collision geometry &rarr; Recast NavMesh &rarr; laptop viewer.", "Callout"))

story.append(P("Photorealistic reconstruction", "H2Custom"))
for text in [
    "MODP was selected because its research focus is monocular online Gaussian reconstruction with detail preservation and global consistency.",
    "The public MODP code expects tracked camera poses and sparse point data. Its public stream loader leaves tracker integration unfinished, so the first adapter will convert COLMAP output into the expected frame packets.",
    "MODP's occupancy structure controls Gaussian distribution; it is not sufficient for physics or navigation.",
    "A fast preview and a later post-refinement pass are planned so the demonstration can show progress before final quality is ready.",
]: story.append(bullet(text))

story.append(P("Obstacle-safe navigation", "H2Custom"))
for text in [
    "The visible Gaussian scene and the hidden collision scene are separate assets.",
    "Dense geometry is filtered, cleaned and simplified before collision use.",
    "Unknown or low-confidence space is blocked. A missing mesh triangle must never become an accidental doorway.",
    "Movement will use a capsule sweep followed by a NavMesh membership check; click-to-move will use Detour pathfinding.",
]: story.append(bullet(text))

story.append(P("Work completed in the repository", "H1Custom"))
for text in [
    "ARCHITECTURE.md: detailed system design, data contracts, coordinate conventions, geometry pipeline, navigation rules, milestones and limitations.",
    "requirements.txt: local Python dependencies for the API, preprocessing, geometry utilities, persistence and verification.",
    ".env: local development configuration with paths, reconstruction settings and conservative navigation parameters.",
    ".env.example: shareable configuration template without machine-specific secrets.",
    ".gitignore: excludes the virtual environment, local environment values, generated data, logs, third-party checkouts and Node output.",
    "scripts/verify_environment.py: import, SQLite, `.env`, bundled FFmpeg and external-tool smoke checks.",
]: story.append(bullet(text))

story.append(PageBreak())
story.append(P("Machine and installation findings", "H1Custom"))
for text in [
    "Windows laptop detected with Python 3.11 and 3.14 available. Python 3.11 was selected for the project virtual environment.",
    "NVIDIA RTX 5060-series laptop GPU detected with approximately 8 GB VRAM.",
    "The virtual environment was created at `.venv`.",
    "The requirements installation was started using the public PyPI index because the default environment index returned no package versions.",
    "All application-layer dependencies were installed from the public PyPI index. Imports, SQLite, .env loading and the bundled FFmpeg 7.1 executable were verified successfully.",
    "COLMAP, FFmpeg, CMake, WSL and `nvcc` were not found on PATH. The NVIDIA driver reports CUDA capability, but that does not mean the CUDA compiler or a MODP-compatible toolkit is installed.",
]: story.append(bullet(text))

story.append(P("Why the environment is split", "H2Custom"))
story.append(P(
    "The local Python environment is intended for the application layer: API endpoints, job metadata, frame inspection, mesh utilities and validation. "
    "MODP's CUDA extensions and the camera-tracking tools are native research components with their own Linux, compiler and GPU constraints. "
    "Keeping those layers separate lets the team validate the application on Windows while testing the mapper in a compatible Linux/CUDA environment.",
    "BodyCustom"
))
story.append(P(
    "The next environment checkpoint is therefore a clean Python import test followed by a separate MODP reference run. "
    "The progress status intentionally does not claim that reconstruction has been executed yet.",
    "Callout"
))

story.append(PageBreak())
story.append(P("Next implementation sequence", "H1Custom"))
steps = [
    ("1", "Record the verified Python baseline", "Environment verification and pip consistency checks passed; preserve resolved package versions before implementation begins."),
    ("2", "Run the MODP reference sequence", "Use a compatible Linux/CUDA environment and one supplied sequence before introducing custom video."),
    ("3", "Build the COLMAP-to-MODP adapter", "Convert images, intrinsics, poses, timestamps and observed sparse points; verify reprojection."),
    ("4", "Process one small indoor video", "Measure registered-frame fraction, splat quality, processing time and GPU memory."),
    ("5", "Build collision geometry", "Generate and review a hidden mesh, scene scale and floor/obstacle confidence."),
    ("6", "Generate and test the NavMesh", "Block unknown space and run wall-penetration, narrow-gap, slope and boundary tests."),
    ("7", "Create the laptop viewer", "Load the scene package, render splats and enable constrained WASD/click-to-move navigation."),
]
step_data = [[P("Step", "Small"), P("Work", "Small"), P("Acceptance check", "Small")]]
for n, work, check in steps:
    step_data.append([P(n, "Small"), P(work, "Small"), P(check, "Small")])
step_tbl = Table(step_data, colWidths=[16*mm, 53*mm, 105*mm], repeatRows=1)
step_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B7285")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E0")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
]))
story.append(step_tbl)

story.append(P("Known risks", "H1Custom"))
for text in [
    "MODP documents a Linux/CUDA 11.8-oriented setup and known issues on Windows/CUDA 12; this machine has a newer driver and no WSL installation.",
    "The laptop's 8 GB VRAM may require shorter videos, lower resolution, fewer keyframes or tighter Gaussian budgets.",
    "Monocular video has scale ambiguity; a known measurement is required before agent dimensions can be treated as metres.",
    "RGB reconstruction cannot guarantee recovery of every real-world obstacle. Conservative unknown-space blocking and manual mesh review are required for the demonstration.",
    "The current PDF records planning and setup progress; no end-to-end reconstruction has been claimed or validated yet.",
]: story.append(bullet(text))

story.append(P("References used for the design", "H1Custom"))
for text in [
    "MODP project and paper: poiw.github.io/MODP and arXiv:2505.07887",
    "MODP repository and public environment notes: github.com/facebookresearch/Online-3DGS-Monocular",
    "COLMAP reconstruction pipeline: github.com/colmap/colmap",
    "gsplat CUDA Gaussian rasterization: github.com/nerfstudio-project/gsplat",
    "Recast/Detour navigation: github.com/recastnavigation/recastnavigation",
    "PlayCanvas and SuperSplat viewer components: developer.playcanvas.com",
]: story.append(bullet(text))

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
    topMargin=16*mm, bottomMargin=18*mm, title="NAMMA progress summary",
    author="NAMMA student project",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
