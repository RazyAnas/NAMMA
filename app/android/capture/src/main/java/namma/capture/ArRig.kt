package namma.capture

import android.app.Activity
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaRecorder
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Build
import android.util.Log
import android.util.Size
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import namma.core.Coverage
import java.io.File
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.acos
import kotlin.math.asin

/**
 * Recording with live coverage (16 Nov): ARCore tracks the phone and measures depth, and the picture ARCore tracks on is
 * what gets recorded. ARCore owns the camera; every frame its camera texture is drawn once to the screen (ScanActivity) and,
 * while recording, once more into the video encoder's surface, full frame, with the camera's own timestamp. The camera pose
 * and the depth image of the same frame go into the coverage map (namma.core.Coverage).
 * (First try, 16 Nov on a Nothing A059: ARCore's shared-camera mode with a MediaRecorder stream beside ARCore's own streams.
 * The camera refused that stream set, "Error configuring streams (-38)". Encoding from the texture needs no extra stream.)
 * What this mode gives up against the plain recorder (CameraRig): ARCore chooses the camera (the main lens, not the
 * ultra-wide), the video is ARCore's picture size (1080p), and exposure stays automatic (tracking needs it). A phone without
 * the Depth API still tracks; the map is then built from ARCore's sparser feature points.
 * frame(), startEncoder / stopEncoder run on the GL thread; start() / stop() on the UI thread only set what the GL thread does.
 */
class ArRig(private val act: Activity, private val onInfo: (String) -> Unit) {
    val coverage = Coverage(0.05f, maxVoxels = 250_000, minHits = 3)     // real depth is noisy: 5 cm, and a voxel counts from its third hit
    private var session: Session? = null
    private var installAsked = false
    private var depthBuf = ShortArray(0); private var confBuf = ByteArray(0)
    private val prevRot = FloatArray(9); private var hasPrev = false
    private var recorder: MediaRecorder? = null
    private var encSurface: EGLSurface? = null
    @Volatile private var wantFile: File? = null; @Volatile private var wantStop = false
    private val imageQuad: FloatBuffer = CoverageGl.buf(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))   // the whole camera image, in the screen quad's order
    private val encUv: FloatBuffer = CoverageGl.buf(FloatArray(8))
    private var clockOffset = Long.MIN_VALUE; private var encoded = 0

    var cameraId = ""; private set
    var size = Size(1920, 1080); private set
    var hevc = false; private set
    var bitrate = 24_000_000; private set
    var depth = false; private set                  // the Depth API is on (else: feature points)
    var hint = 0
    @Volatile var arActive = false; private set
    @Volatile var recording = false; private set
    @Volatile var tracking = false; private set
    val view = FloatArray(16); val proj = FloatArray(16); val pose = FloatArray(16)
    /** Degrees the camera turned since [takeTurn] was last called, and where it aims (degrees above the horizon). */
    @Volatile private var turned = 0.0
    @Volatile var elevation = Double.NaN; private set

    /** null = ARCore is ready; else what to tell the user (not installed yet, phone not supported, ...). */
    fun unavailable(): String? = try {
        when (ArCoreApk.getInstance().requestInstall(act, !installAsked)) {
            ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installAsked = true; "Installing Google Play Services for AR: come back when it is done" }
            else -> null
        }
    } catch (e: UnavailableException) { "Live coverage needs ARCore, which this phone cannot run (${e.javaClass.simpleName.removePrefix("Unavailable").removeSuffix("Exception")})" }
    catch (e: Exception) { "ARCore: ${e.message}" }

    /** Start ARCore on the camera. cameraTexture = the GL texture the picture goes to (CoverageGl.cameraTexture). Throws with
     *  a readable message if ARCore cannot start. */
    fun open(cameraTexture: Int) {
        val s = session ?: Session(act).also { session = it }
        // 30 fps, the largest picture ARCore offers up to 1920 wide: that picture is the video
        val configs = s.getSupportedCameraConfigs(CameraConfigFilter(s).setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30)))
        configs.filter { it.textureSize.width <= 1920 }.maxByOrNull { it.textureSize.width * it.textureSize.height }?.let { s.cameraConfig = it }
        val cfg = Config(s); cfg.focusMode = Config.FocusMode.AUTO; cfg.updateMode = Config.UpdateMode.BLOCKING        // one draw = one new camera frame
        depth = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC); cfg.depthMode = if (depth) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        cfg.planeFindingMode = Config.PlaneFindingMode.DISABLED; cfg.lightEstimationMode = Config.LightEstimationMode.DISABLED
        s.configure(cfg); cameraId = s.cameraConfig.cameraId; s.setCameraTextureName(cameraTexture)
        val t = s.cameraConfig.textureSize; size = Size(t.width - t.width % 2, t.height - t.height % 2)
        fun cap(mime: String) = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder && mime in it.supportedTypes }
            .mapNotNull { runCatching { it.getCapabilitiesForType(mime).videoCapabilities }.getOrNull() }.firstOrNull { it.isSizeSupported(size.width, size.height) }
        val h = cap(MediaFormat.MIMETYPE_VIDEO_HEVC); val a = cap(MediaFormat.MIMETYPE_VIDEO_AVC)
        hevc = h != null; bitrate = minOf(60_000_000, (h ?: a)?.bitrateRange?.upper ?: 24_000_000)
        s.resume(); arActive = true
    }

    val sensorOrientation: Int get() = try {
        act.getSystemService(CameraManager::class.java)!!.getCameraCharacteristics(cameraId).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
    } catch (e: Exception) { 90 }

    /**
     * One frame, on the GL thread: ARCore's newest picture goes to the camera texture, [view] / [proj] / [pose] are set; while
     * recording the picture is also drawn into the encoder and the depth image (or the feature points) fused into [coverage].
     * Returns the frame while ARCore runs. Never throws.
     */
    fun frame(gl: CoverageGl, w: Int, h: Int, rotation: Int): Frame? {
        val s = session ?: return null
        if (!arActive) return null
        val f = try { s.setDisplayGeometry(rotation, w, h); s.update() } catch (e: Exception) { return null }
        try {
            if (f.hasDisplayGeometryChanged()) f.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, gl.quadNdc, Coordinates2d.TEXTURE_NORMALIZED, gl.uv)
            wantFile?.let { wantFile = null; startEncoder(it) }
            if (wantStop) { wantStop = false; stopEncoder() }
            if (recording) encode(f, gl, w, h)
            val cam = f.camera; tracking = cam.trackingState == TrackingState.TRACKING
            if (!tracking) return f
            cam.getViewMatrix(view, 0); cam.getProjectionMatrix(proj, 0, 0.05f, 60f); cam.pose.toMatrix(pose, 0)
            motion()
            if (recording) fuse(f)
        } catch (e: Exception) { onInfo("Live coverage: ${e.message}") }
        return f
    }

    // ---- the video: ARCore's camera texture drawn into the encoder (GL thread)
    private fun startEncoder(file: File) {
        try {
            val m = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(act) else @Suppress("DEPRECATION") MediaRecorder()
            m.setVideoSource(MediaRecorder.VideoSource.SURFACE); m.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4); m.setOutputFile(file.path)
            m.setVideoEncodingBitRate(bitrate); m.setVideoFrameRate(30); m.setVideoSize(size.width, size.height)
            m.setVideoEncoder(if (hevc) MediaRecorder.VideoEncoder.HEVC else MediaRecorder.VideoEncoder.H264)
            m.setOrientationHint(hint); m.prepare()
            val dpy = EGL14.eglGetCurrentDisplay(); val ctx = EGL14.eglGetCurrentContext()
            val id = IntArray(1); EGL14.eglQueryContext(dpy, ctx, EGL14.EGL_CONFIG_ID, id, 0)        // the screen's own config (ScanActivity asks for a recordable one)
            val cfgs = arrayOfNulls<EGLConfig>(1); val n = IntArray(1)
            EGL14.eglChooseConfig(dpy, intArrayOf(EGL14.EGL_CONFIG_ID, id[0], EGL14.EGL_NONE), 0, cfgs, 0, 1, n, 0)
            val es = EGL14.eglCreateWindowSurface(dpy, cfgs[0], m.surface, intArrayOf(EGL14.EGL_NONE), 0)
            check(es != null && es != EGL14.EGL_NO_SURFACE) { "no encoder surface (EGL error 0x${Integer.toHexString(EGL14.eglGetError())})" }
            m.start(); recorder = m; encSurface = es; clockOffset = Long.MIN_VALUE; encoded = 0
            coverage.clear(); hasPrev = false; turned = 0.0; recording = true
            Log.i(TAG, "recording ${size.width}x${size.height} ${if (hevc) "hevc" else "h264"} ${bitrate / 1_000_000} Mbps -> ${file.name}")
        } catch (e: Exception) { Log.e(TAG, "encoder start failed", e); recorder?.release(); recorder = null; encSurface = null; recording = false; onInfo("Could not start recording: ${e.message}") }
    }

    private fun encode(f: Frame, gl: CoverageGl, w: Int, h: Int) {
        val es = encSurface ?: return
        val dpy = EGL14.eglGetCurrentDisplay(); val ctx = EGL14.eglGetCurrentContext()
        val draw = EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW); val read = EGL14.eglGetCurrentSurface(EGL14.EGL_READ)
        f.transformCoordinates2d(Coordinates2d.IMAGE_NORMALIZED, imageQuad, Coordinates2d.TEXTURE_NORMALIZED, encUv)   // the whole picture, not the screen's crop
        if (!EGL14.eglMakeCurrent(dpy, es, es, ctx)) { if (encoded == 0) Log.e(TAG, "encoder surface not current: 0x${Integer.toHexString(EGL14.eglGetError())}"); return }
        GLES20.glViewport(0, 0, size.width, size.height); gl.background(encUv)
        // the camera's own frame times, moved onto the clock the recorder started on (System.nanoTime): MediaRecorder drops every
        // frame stamped before its start, and the camera's clock is a different one (first phone test: 'The recording is empty')
        if (clockOffset == Long.MIN_VALUE) clockOffset = System.nanoTime() - f.timestamp
        EGLExt.eglPresentationTimeANDROID(dpy, es, f.timestamp + clockOffset)
        if (!EGL14.eglSwapBuffers(dpy, es) && encoded == 0) Log.e(TAG, "encoder swap failed: 0x${Integer.toHexString(EGL14.eglGetError())}")
        encoded++
        EGL14.eglMakeCurrent(dpy, draw, read, ctx); GLES20.glViewport(0, 0, w, h)
    }

    private fun stopEncoder() {
        if (!recording) return
        recording = false
        try { recorder?.stop(); Log.i(TAG, "recording closed: $encoded frames") } catch (e: RuntimeException) { Log.e(TAG, "recorder stop failed after $encoded frames", e); onInfo("The recording was too short") }
        recorder?.release(); recorder = null
        encSurface?.let { EGL14.eglDestroySurface(EGL14.eglGetCurrentDisplay(), it) }; encSurface = null
        stopped?.invoke()
    }

    /** Turn since the last frame (the angle between the two camera orientations) and where the camera aims. */
    private fun motion() {
        val m = pose
        if (hasPrev) {
            var tr = 0f
            for (c in 0 until 3) for (r in 0 until 3) tr += prevRot[3 * c + r] * m[4 * c + r]           // trace(Rprev^T R)
            turned += Math.toDegrees(acos(((tr - 1f) / 2f).coerceIn(-1f, 1f).toDouble()))
        }
        for (c in 0 until 3) for (r in 0 until 3) prevRot[3 * c + r] = m[4 * c + r]
        hasPrev = true; elevation = Math.toDegrees(asin((-m[9]).coerceIn(-1f, 1f).toDouble()))            // forward = -z axis of the pose
    }

    fun takeTurn(): Double { val t = turned; turned = 0.0; return t }

    private fun fuse(f: Frame) {
        if (depth) {
            try {
                // raw depth + its confidence: only what ARCore actually measured (the smoothed image fills gaps with guesses, and
                // guesses far away painted whole walls that were never seen: first phone test, 16 Nov)
                f.acquireRawDepthImage16Bits().use { img -> f.acquireRawDepthConfidenceImage().use { ci ->
                    val w = img.width; val h = img.height; val p = img.planes[0]; val sb = p.buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
                    if (depthBuf.size != w * h) { depthBuf = ShortArray(w * h); confBuf = ByteArray(w * h) }
                    val stride = p.rowStride / 2
                    for (y in 0 until h) { sb.position(y * stride); sb.get(depthBuf, y * w, w) }
                    val cp = ci.planes[0]; val cb = cp.buffer
                    if (ci.width == w && ci.height == h) for (y in 0 until h) { cb.position(y * cp.rowStride); cb.get(confBuf, y * w, w) } else confBuf.fill(-1)
                    val k = f.camera.imageIntrinsics; val dim = k.imageDimensions; val fl = k.focalLength; val pp = k.principalPoint
                    val kx = w / dim[0].toFloat(); val ky = h / dim[1].toFloat()             // the depth image is the camera image, smaller
                    coverage.addDepth(depthBuf, w, h, fl[0] * kx, fl[1] * ky, pp[0] * kx, pp[1] * ky, pose, stride = 2, maxM = 3.5f, confidence = confBuf, minConfidence = 128)
                } }
            } catch (e: NotYetAvailableException) {} catch (e: Exception) {}                  // the first frames have no depth yet
        } else {
            try {
                f.acquirePointCloud().use { pc ->
                    val b = pc.points; val n = b.remaining() / 4                              // x, y, z, confidence
                    for (i in 0 until n) if (b.get(4 * i + 3) > 0.3f) coverage.add(b.get(4 * i), b.get(4 * i + 1), b.get(4 * i + 2), pose[12], pose[13], pose[14])
                }
            } catch (e: Exception) {}
        }
    }

    /** The camera picture's brightness plane, shrunk to about a quarter (for the blur / dark checks), or null. */
    fun luma(f: Frame): Triple<IntArray, Int, Int>? = try {
        f.acquireCameraImage().use { img ->
            val p = img.planes[0]; val step = maxOf(1, img.width / 320); val w = img.width / step; val h = img.height / step
            val b = p.buffer; val out = IntArray(w * h)
            for (y in 0 until h) for (x in 0 until w) out[y * w + x] = b.get(y * step * p.rowStride + x * step * p.pixelStride).toInt() and 0xFF
            Triple(out, w, h)
        }
    } catch (e: Exception) { null }

    /** Called (GL thread) when the video file is closed after [stop]. */
    @Volatile var stopped: (() -> Unit)? = null

    fun start(file: File) { wantStop = false; wantFile = file }

    fun stop() { wantFile = null; if (recording) wantStop = true }

    /** Leaving the screen (GL thread, before the surface goes): close the video now. */
    fun stopNow() { wantFile = null; wantStop = false; stopEncoder() }

    fun close() {
        arActive = false; tracking = false
        try { session?.pause() } catch (e: Exception) {}
    }

    fun quit() { close(); try { session?.close() } catch (e: Exception) {}; session = null }

    private companion object { const val TAG = "NammaAr" }
}
