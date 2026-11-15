package namma.capture

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.StatFs
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import namma.core.Coach
import namma.core.Coverage
import namma.core.CoverageDemo
import namma.core.Jobs
import namma.core.Quality
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Recording with live coverage (16 Nov): the camera picture with the scanned surfaces painted on it while you film:
 * red = seen from one direction (come back to it), yellow = two, green = three or more (enough for the model); no paint =
 * not filmed yet. Top right: the same from above, with you as the white dot. The numbers say how much of what was seen is
 * covered well, and the tip line points at the nearest red area. ARCore does the tracking and the depth (ArRig), the map is
 * namma.core.Coverage, the drawing CoverageGl. After STOP: the same video check, name and upload as the normal recorder.
 * Intent extra "demo" = true: a made-up room instead of the camera (CoverageDemo), to see the screen work without ARCore.
 */
class ScanActivity : Activity(), GLSurfaceView.Renderer {
    private val demo by lazy { intent.getBooleanExtra("demo", false) }
    private lateinit var glView: GLSurfaceView
    private val gl = CoverageGl()
    private var rig: ArRig? = null
    private val sim = CoverageDemo(); private val simMap = Coverage(); private var simT0 = 0L; @Volatile private var simRec = false
    private lateinit var title: TextView; private lateinit var timer: TextView; private lateinit var cue: TextView
    private lateinit var tip: TextView; private lateinit var info: TextView; private lateinit var recBtn: android.widget.Button
    private lateinit var mini: ImageView
    private var tts: TextToSpeech? = null
    private val coach = Coach()
    private var job = ""; private var file: File? = null; private var startedAt = 0L
    private val worker = HandlerThread("scan-check").apply { start() }
    private val work = Handler(worker.looper)
    private var extractor = Quality.Extractor(); private val samples = ArrayList<Quality.Sample>()
    private val track = JSONArray()                                                    // camera position twice a second (meta.json)
    private var vw = 1; private var vh = 1; private var frames = 0
    private var lastSample = 0L; private var lastHud = 0L; private var lastFrameAt = 0L; private var turnForSample = 0.0
    private val vp = FloatArray(16); private val view = FloatArray(16); private val proj = FloatArray(16)
    @Volatile private var surfaceReady = false; private var opened = false; private var lastCueAt = 0L
    @Volatile private var lastStats: Coverage.Stats? = null
    @Volatile private var recWanted = false                                            // REC was tapped (the encoder starts a frame later)
    private val recording get() = if (demo) simRec else recWanted

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) tts?.language = Locale.UK }
        if (!demo) rig = ArRig(this) { m -> runOnUiThread { say(m, speak = false) } }
        build()
        if (!demo && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) openCamera() else { say("The camera permission is needed", false); finish() }
    }

    private fun build() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        val dm = resources.displayMetrics; val sw = maxOf(dm.widthPixels, dm.heightPixels); val sh = minOf(dm.widthPixels, dm.heightPixels)
        val pw = minOf(sw, sh * 16 / 9); val ph = pw * 9 / 16                         // landscape 16:9 picture, centred
        glView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2); preserveEGLContextOnPause = true
            setEGLConfigChooser { egl, display ->                                    // a config the video encoder's surface can share (ArRig.startEncoder)
                val want = intArrayOf(EGL10.EGL_RED_SIZE, 8, EGL10.EGL_GREEN_SIZE, 8, EGL10.EGL_BLUE_SIZE, 8, EGL10.EGL_ALPHA_SIZE, 8, EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_RENDERABLE_TYPE, 4, 0x3142, 1, EGL10.EGL_NONE)
                val out = arrayOfNulls<EGLConfig>(1); val n = IntArray(1)                 // 4 = OpenGL ES 2, 0x3142 = EGL_RECORDABLE_ANDROID
                if (!egl.eglChooseConfig(display, want, out, 1, n) || n[0] == 0)
                    egl.eglChooseConfig(display, intArrayOf(EGL10.EGL_RED_SIZE, 8, EGL10.EGL_GREEN_SIZE, 8, EGL10.EGL_BLUE_SIZE, 8, EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_RENDERABLE_TYPE, 4, EGL10.EGL_NONE), out, 1, n)
                out[0]
            }
            setRenderer(this@ScanActivity); renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        root.addView(glView, FrameLayout.LayoutParams(pw, ph, Gravity.CENTER))

        val top = Ui.column(this, 8).apply { setBackgroundColor(0x99000000.toInt()) }
        title = Ui.text(this, if (demo) "Live coverage: demo room" else "Live coverage", 18f, bold = true); timer = Ui.text(this, "Starting…", 14f, Ui.DIM)
        top.addView(title); top.addView(timer)
        val legend = Ui.row(this)
        for ((c, s) in listOf(CoverageGl.COLOUR_INT[1] to "one side", CoverageGl.COLOUR_INT[2] to "two", CoverageGl.COLOUR_INT[3] to "covered")) {
            legend.addView(View(this).apply { setBackgroundColor(c) }, LinearLayout.LayoutParams(Ui.dp(this, 12), Ui.dp(this, 12)).apply { rightMargin = Ui.dp(this@ScanActivity, 5) })
            legend.addView(Ui.text(this, s, 12f, Ui.DIM), LinearLayout.LayoutParams(-2, -2).apply { rightMargin = Ui.dp(this@ScanActivity, 14) })
        }
        top.addView(legend)
        root.addView(top, FrameLayout.LayoutParams(sw * 50 / 100, -2, Gravity.TOP or Gravity.START))

        cue = Ui.text(this, "", 22f, Color.WHITE, true).apply { gravity = Gravity.CENTER; setBackgroundColor(0xCCE5484D.toInt()); visibility = View.GONE; setPadding(24, 16, 24, 16) }
        root.addView(cue, FrameLayout.LayoutParams(sw * 60 / 100, -2, Gravity.CENTER))

        val bottom = Ui.column(this, 8).apply { setBackgroundColor(0x99000000.toInt()) }
        tip = Ui.text(this, "Stand at the entrance and tap REC. Walk slowly along the walls; paint everything green.", 13f); info = Ui.text(this, "", 11f, Ui.DIM)
        bottom.addView(tip); bottom.addView(info)
        root.addView(bottom, FrameLayout.LayoutParams(sw * 62 / 100, -2, Gravity.BOTTOM or Gravity.START))

        val col = Ui.column(this, 8).apply { gravity = Gravity.CENTER_HORIZONTAL }      // the map from above, then the button under the thumb
        mini = ImageView(this).apply { setBackgroundColor(0x99000000.toInt()); scaleType = ImageView.ScaleType.FIT_CENTER }
        col.addView(mini, LinearLayout.LayoutParams(Ui.dp(this, 132), Ui.dp(this, 132)))
        col.addView(View(this), LinearLayout.LayoutParams(1, 0, 1f))
        recBtn = Ui.button(this, "●  REC", Ui.WARN) { if (recording) stopRec() else startRec() }
        col.addView(recBtn, LinearLayout.LayoutParams(Ui.dp(this, 120), -2).apply { setMargins(0, 8, 0, 8) })
        col.addView(View(this), LinearLayout.LayoutParams(1, 0, 1f))
        root.addView(col, FrameLayout.LayoutParams(-2, -1, Gravity.END))
        recBtn.isEnabled = demo; recBtn.alpha = if (demo) 1f else 0.4f
        Ui.insets(root, top, bottom, col)
        setContentView(root)
    }

    @Suppress("DEPRECATION")
    private fun rotation() = windowManager.defaultDisplay.rotation
    private fun rotationDeg() = when (rotation()) { Surface.ROTATION_90 -> 90; Surface.ROTATION_180 -> 180; Surface.ROTATION_270 -> 270; else -> 0 }

    private fun openCamera() {
        val r = rig ?: return
        if (opened || !surfaceReady || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        r.unavailable()?.let { timer.text = it; tip.text = "$it. The normal recorder (Record a room) works without it."; return }
        opened = true
        try {
            r.open(gl.cameraTexture); r.stopped = { runOnUiThread { afterStop() } }
            run {
                r.hint = (r.sensorOrientation - rotationDeg() + 360) % 360; recBtn.isEnabled = true; recBtn.alpha = 1f
                info.text = "main lens · ${r.size.width}x${r.size.height} ${if (r.hevc) "HEVC" else "H.264"} ${r.bitrate / 1_000_000} Mbps · 30 fps · " +
                    "stabilisation off · exposure automatic · ${if (r.depth) "depth from ARCore" else "no depth on this phone: coverage from feature points (sparser)"}"
            }
        } catch (e: Exception) { opened = false; timer.text = "ARCore: ${e.message}"; say("Live coverage could not start: ${e.message}", false) }
    }

    // ---- drawing (GL thread)
    override fun onSurfaceCreated(g: GL10?, c: EGLConfig?) { gl.create(); surfaceReady = true; runOnUiThread { openCamera() } }

    override fun onSurfaceChanged(g: GL10?, w: Int, h: Int) { vw = w; vh = h; GLES20.glViewport(0, 0, w, h) }

    override fun onDrawFrame(g: GL10?) {
        val now = SystemClock.elapsedRealtime(); val dt = if (lastFrameAt == 0L) 0.033 else (now - lastFrameAt) / 1000.0; lastFrameAt = now; frames++
        GLES20.glClearColor(0.05f, 0.06f, 0.08f, 1f); GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val map: Coverage; val pose: FloatArray; var tracking = true
        if (demo) {
            if (simT0 == 0L) simT0 = now
            sim.step((now - simT0) / 1000.0); pose = sim.pose
            if (simRec) simMap.addDepth(sim.depthMm, sim.w, sim.h, sim.fx, sim.fy, sim.px, sim.py, pose, stride = 1)
            invertRigid(pose, view); perspective(proj, 2f * sim.fx / sim.w, 2f * sim.fy / sim.h, 0.05f, 60f); map = simMap
        } else {
            val r = rig ?: return; val f = r.frame(gl, vw, vh, rotation()) ?: return
            gl.background(); tracking = r.tracking; map = r.coverage; pose = r.pose
            if (!tracking) { if (now - lastHud > 500) { lastHud = now; runOnUiThread { timer.text = "Finding where the phone is: move it slowly sideways" } }; return }
            System.arraycopy(r.view, 0, view, 0, 16); System.arraycopy(r.proj, 0, proj, 0, 16)
            val turn = r.takeTurn(); turnForSample += turn
            if (recWanted && r.recording) {
                coach.gyro(now, 0f, Math.toRadians(turn / maxOf(dt, 1e-3)).toFloat(), 0f)?.let { c -> runOnUiThread { show(c) } }
                if (now - lastSample >= 250) {                                         // 4 a second: the video check's samples + blur / dark cues
                    lastSample = now; val t = (now - startedAt) / 1000.0; val tn = turnForSample; turnForSample = 0.0; val el = r.elevation
                    r.luma(f)?.let { (l, w, h) -> work.post {
                        val smp = extractor.add(t, l, w, h, tn, el); synchronized(samples) { samples.add(smp) }
                        val e = Math.toRadians(el); val c = coach.frame(now, smp.luma, smp.sharp) ?: coach.gravity(now, 0f, (9.81 * Math.cos(e)).toFloat(), (-9.81 * Math.sin(e)).toFloat())
                        if (c != null) runOnUiThread { show(c) }
                    } }
                }
            }
        }
        if (frames % 6 == 0) gl.upload(map.snapshot())                                 // the map changes slowly: 5 uploads a second
        Matrix.multiplyMM(vp, 0, proj, 0, view, 0); gl.coverage(vp, proj[0] * vw / 2f)
        if (now - lastHud > 500) { lastHud = now; hud(map, pose, now) }
    }

    /** Numbers, the map from above and the "go there" tip, twice a second (built on the GL thread, shown on the UI thread). */
    private fun hud(map: Coverage, pose: FloatArray, now: Long) {
        val st = map.stats(); lastStats = st; val rec = recording
        val x = pose[12]; val y = pose[13]; val z = pose[14]
        if (rec) track.put(JSONArray(listOf(r3((now - startedAt) / 1000.0), r3(x.toDouble()), r3(y.toDouble()), r3(z.toDouble()))))
        val weak = if (rec) map.weakest(x, y, z) else null
        val where = weak?.let {
            val fx = -pose[8]; val fz = -pose[10]; val dx = it.x - x; val dz = it.z - z                 // camera forward and the way to the spot, on the floor
            val a = Math.toDegrees(atan2((fx * dz - fz * dx).toDouble(), (fx * dx + fz * dz).toDouble())); val d = sqrt(dx * dx + dz * dz)
            val side = when { a > 135 || a < -135 -> "behind you"; a > 25 -> "to your right"; a < -25 -> "to your left"; else -> "ahead" }
            "Red area %.1f m %s: film it from another side.".format(d, side)
        }
        val bmp = map.topDown(0.1f)?.let { m ->
            val b = Bitmap.createBitmap(m.w, m.h, Bitmap.Config.ARGB_8888); val px = IntArray(m.w * m.h) { CoverageGl.COLOUR_INT[m.level[it].toInt()] }
            val cx = ((x - m.x0) / m.cell).toInt(); val cz = ((z - m.z0) / m.cell).toInt()
            for (dz in -1..1) for (dx in -1..1) { val i = (cz + dz) * m.w + cx + dx; if (cx + dx in 0 until m.w && cz + dz in 0 until m.h) px[i] = Color.WHITE }
            b.setPixels(px, 0, m.w, 0, 0, m.w, m.h); b
        }
        runOnUiThread {
            val s = if (rec) (SystemClock.elapsedRealtime() - startedAt) / 1000 else 0
            timer.text = if (!rec) "Ready. Tap REC and walk." else
                "REC  %d:%02d   ·   covered well %d%%   ·   %s to revisit".format(s / 60, s % 60, st.percent, area(st.weak, st.voxelM))
            if (rec && where != null && SystemClock.elapsedRealtime() - lastCueAt > 2500) tip.text = where
            else if (rec && where == null && st.seen > 500) tip.text = if (st.percent >= 80) "Well covered. Move on to what is not painted yet, or stop." else "Keep going: walk past each wall and look back at it from the other side."
            bmp?.let { mini.setImageDrawable(BitmapDrawable(resources, it).apply { isFilterBitmap = false }) }
            if (SystemClock.elapsedRealtime() - lastCueAt > 2500) cue.visibility = View.GONE
            if (!demo && recWanted && rig?.recording != true && SystemClock.elapsedRealtime() - startedAt > 3000) {   // the encoder never started
                recWanted = false; recBtn.text = "●  REC"; file?.delete(); file = null
            }
        }
    }

    private fun area(voxels: Int, voxelM: Float) = "%.1f m²".format(voxels * voxelM * voxelM)
    private fun r3(v: Double) = Math.round(v * 1000) / 1000.0

    // ---- recording
    private fun startRec() {
        if (demo) { simMap.clear(); simRec = true; startedAt = SystemClock.elapsedRealtime(); recBtn.text = "■  STOP"; return }
        val r = rig ?: return
        if (!r.tracking) { say("Wait until the phone knows where it is (move it slowly sideways)", false); return }
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        if (StatFs(dir.path).availableBytes < 2L shl 30) { say("Less than 2 GB free on the phone: free some space first", false); return }
        job = Jobs.newId(); file = File(dir, "$job.mp4")
        r.start(file!!); recWanted = true
        startedAt = SystemClock.elapsedRealtime(); synchronized(samples) { samples.clear() }; extractor = Quality.Extractor()
        while (track.length() > 0) track.remove(0)
        recBtn.text = "■  STOP"; say("Recording. Walk slowly along the walls.", true, warn = false)
    }

    private fun stopRec() {
        if (demo) { simRec = false; recBtn.text = "●  REC"; lastStats?.let { say("Demo: ${it.percent}% of what was seen is covered well", false, warn = false) }; return }
        val r = rig ?: return; if (!recWanted) return
        recWanted = false; stoppedAt = SystemClock.elapsedRealtime(); recBtn.text = "●  REC"; cue.visibility = View.GONE
        if (r.recording) r.stop() else { r.stop(); afterStop() }                       // the encoder never started: nothing to wait for
    }

    private var stoppedAt = 0L

    /** The video file is closed (ArRig.stopped): check it, then name + upload. */
    private fun afterStop() {
        val f = file ?: return; file = null; val seconds = (stoppedAt - startedAt) / 1000; val st = lastStats
        if (!f.exists() || f.length() < 100_000) { say("The recording is empty, try again", false); f.delete(); return }
        work.post {
            val list = synchronized(samples) { ArrayList(samples) }; val q = Quality.score(list, locked = true)   // exposure is automatic here by design
            runOnUiThread { checked(f, seconds, q, list, st) }
        }
    }

    private fun checked(f: File, seconds: Long, q: Quality.Report, list: List<Quality.Sample>, st: Coverage.Stats?) {
        val word = when (q.verdict) { Quality.GOOD -> "Good"; Quality.RISKY -> "Risky"; else -> "Bad" }
        val box = Ui.column(this, 20)
        box.addView(Ui.text(this, "Video check: $word (${q.score}/100)", 20f, if (q.good) Ui.ACCENT else Ui.WARN, true))
        if (st != null) box.addView(Ui.text(this, "Coverage: ${st.percent}% of the filmed surface was seen from three or more sides; " +
            "${area(st.weak, st.voxelM)} from one side only.", 15f, if (st.percent >= 60) Ui.TEXT else Ui.WARN))
        box.addView(Ui.text(this, q.words(), 14f, Ui.DIM))
        val name = EditText(this).apply { setText("Room ${java.text.SimpleDateFormat("d MMM HH:mm", Locale.getDefault()).format(java.util.Date())}"); setSelectAllOnFocus(true); setTextColor(Ui.TEXT) }
        box.addView(Ui.text(this, "Name (${seconds / 60} min ${seconds % 60} s)", 14f, Ui.DIM)); box.addView(name)
        val nm = { name.text.toString().ifBlank { "Room" } }
        val d = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(box) }).setCancelable(false)
        if (q.verdict == Quality.BAD) d.setPositiveButton("Record again") { _, _ -> f.delete(); say("Deleted. Tap REC when ready.", true, warn = false) }
            .setNeutralButton("Upload anyway") { _, _ -> save(f, nm(), seconds, q, list, st, forced = true) }
        else d.setPositiveButton("Upload") { _, _ -> save(f, nm(), seconds, q, list, st, forced = false) }
            .setNeutralButton("Delete") { _, _ -> f.delete() }
        if (!isFinishing && !isDestroyed) d.show()
    }

    private fun save(f: File, name: String, seconds: Long, q: Quality.Report, list: List<Quality.Sample>, st: Coverage.Stats?, forced: Boolean) {
        val r = rig ?: return
        val j = LocalJob(job, name, f.path, System.currentTimeMillis()).apply { total = f.length() }
        val meta = JSONObject().put("name", name).put("preset", "best").put("seconds", seconds).put("rooms", 1)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL} (Android ${Build.VERSION.RELEASE})")
            .put("camera", JSONObject().put("mode", "ar").put("lens", "main").put("zoom", 1.0).put("width", r.size.width).put("height", r.size.height)
                .put("fps", 30).put("bitrate", r.bitrate).put("codec", if (r.hevc) "hevc" else "h264").put("locked", false)
                .put("exposure", "automatic (ARCore)").put("manual_shutter", false).put("stabilisation", "off").put("orientation", "landscape"))
            .put("cues", JSONObject(coach.counts.mapKeys { it.key.name }.mapValues { it.value }))
            .put("coverage", JSONObject().put("depth", r.depth).put("seen_voxels", st?.seen ?: 0).put("good_voxels", st?.good ?: 0).put("weak_voxels", st?.weak ?: 0)
                .put("percent", st?.percent ?: 0).put("voxel_m", (st?.voxelM ?: 0f).toDouble()).put("track_columns", JSONArray(listOf("t", "x", "y", "z"))).put("track", track))
            .put("quality", q.json().put("uploaded_anyway", forced))
            .put("quality_samples", JSONObject().put("columns", JSONArray(Quality.COLUMNS)).put("rows", JSONArray(list.map { it.json() })))
        j.meta.writeText(meta.toString(1))
        Store.add(this, j); UploadService.start(this); CheckJob.schedule(this)
        say("Saved. Uploading in the background; you can close the app.", true, warn = false)
        glView.postDelayed({ finish() }, 1500)
    }

    private fun show(c: Coach.Cue) = say(c.text, speak = true, warn = true)

    private fun say(text: String, speak: Boolean, warn: Boolean = true) {
        cue.text = text; cue.setBackgroundColor(if (warn) 0xCCE5484D.toInt() else 0xCC2BB3A3.toInt()); cue.visibility = View.VISIBLE; lastCueAt = SystemClock.elapsedRealtime()
        if (speak) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue")
    }

    override fun onResume() { super.onResume(); glView.onResume(); openCamera() }

    override fun onPause() {
        if (recording && !demo) {                                      // leaving mid-recording: keep what was filmed. The video is closed on
            recWanted = false; stoppedAt = SystemClock.elapsedRealtime(); recBtn.text = "●  REC"   // the GL thread, before its surface goes
            glView.queueEvent { rig?.stopNow() }
        } else if (recording) stopRec()
        glView.onPause(); rig?.close(); opened = false; recBtn.isEnabled = demo; recBtn.alpha = if (demo) 1f else 0.4f
        super.onPause()
    }

    override fun onDestroy() { rig?.quit(); tts?.shutdown(); worker.quitSafely(); super.onDestroy() }

    companion object {
        /** The inverse of a rigid camera-to-world matrix (column-major): the view matrix. */
        fun invertRigid(m: FloatArray, o: FloatArray) {
            o[0] = m[0]; o[1] = m[4]; o[2] = m[8]; o[3] = 0f; o[4] = m[1]; o[5] = m[5]; o[6] = m[9]; o[7] = 0f; o[8] = m[2]; o[9] = m[6]; o[10] = m[10]; o[11] = 0f
            o[12] = -(m[0] * m[12] + m[1] * m[13] + m[2] * m[14]); o[13] = -(m[4] * m[12] + m[5] * m[13] + m[6] * m[14]); o[14] = -(m[8] * m[12] + m[9] * m[13] + m[10] * m[14]); o[15] = 1f
        }

        /** Perspective projection from the focal lengths as a share of the half picture (sx = 2 fx / width, sy = 2 fy / height). */
        fun perspective(o: FloatArray, sx: Float, sy: Float, near: Float, far: Float) {
            o.fill(0f); o[0] = sx; o[5] = sy; o[10] = -(far + near) / (far - near); o[11] = -1f; o[14] = -2f * far * near / (far - near)
        }
    }
}
