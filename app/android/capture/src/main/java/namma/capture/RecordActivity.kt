package namma.capture

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.StatFs
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.view.Gravity
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import namma.core.Coach
import namma.core.Guide
import namma.core.Jobs
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** Guided recording: checklist -> LOCK on the brightest wall -> record with pass prompts (lap, reverse lap, diagonals, doorway)
 *  and live cues from the gyroscope (turn speed), gravity (aim) and the preview picture (blur, darkness, light changes). */
class RecordActivity : Activity(), SensorEventListener {
    private lateinit var rig: CameraRig
    private lateinit var texture: TextureView
    private lateinit var passTitle: TextView; private lateinit var timer: TextView; private lateinit var cue: TextView
    private lateinit var tip: TextView; private lateinit var info: TextView
    private lateinit var lockBtn: View; private lateinit var recBtn: android.widget.Button; private lateinit var nextBtn: View
    private lateinit var checklist: View
    private var tts: TextToSpeech? = null
    private val coach = Coach()
    private var job = ""; private var file: File? = null
    private var startedAt = 0L; private var pass = 0; private var room = 1
    private val passLog = JSONArray()
    private var lastCueAt = 0L
    private val frame = Bitmap.createBitmap(72, 128, Bitmap.Config.ARGB_8888)
    private val px = IntArray(72 * 128); private val luma = IntArray(72 * 128)
    private var camReady = false

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        rig = CameraRig(this) { m -> runOnUiThread { say(m, speak = false) } }
        tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) tts?.language = Locale.UK }
        build()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (res.firstOrNull() == PackageManager.PERMISSION_GRANTED) openCamera() else { say("The camera permission is needed", false); finish() }
    }

    private fun build() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        val w = resources.displayMetrics.widthPixels
        texture = TextureView(this)
        root.addView(texture, FrameLayout.LayoutParams(w, w * 16 / 9, Gravity.TOP))         // 1080p portrait: 9:16
        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, a: Int, b: Int) = openCamera()
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, a: Int, b: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
        }

        val top = Ui.column(this, 12).apply { setBackgroundColor(0x99000000.toInt()) }
        passTitle = Ui.text(this, "Before you start", 20f, bold = true); timer = Ui.text(this, "", 15f, Ui.DIM)
        top.addView(passTitle); top.addView(timer)
        root.addView(top, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        cue = Ui.text(this, "", 24f, Color.WHITE, true).apply {
            gravity = Gravity.CENTER; setBackgroundColor(0xCCE5484D.toInt()); visibility = View.GONE; setPadding(24, 24, 24, 24)
        }
        root.addView(cue, FrameLayout.LayoutParams(-1, -2, Gravity.CENTER))

        val bottom = Ui.column(this, 12).apply { setBackgroundColor(0x99000000.toInt()) }
        tip = Ui.text(this, Guide.LOCK, 15f); info = Ui.text(this, "", 12f, Ui.DIM)
        bottom.addView(tip); bottom.addView(info)
        val row = Ui.row(this)
        lockBtn = Ui.button(this, "LOCK", Ui.CARD) { toggleLock() }
        recBtn = Ui.button(this, "●  REC", Ui.WARN) { if (rig.recording) stopRec() else startRec() }
        nextBtn = Ui.button(this, "Next ▶", Ui.ACCENT) { nextPass() }
        for (v in listOf(lockBtn, recBtn, nextBtn)) row.addView(v, LinearLayout.LayoutParams(0, -2, 1f).apply { setMargins(6, 6, 6, 6) })
        bottom.addView(row)
        root.addView(bottom, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        recBtn.isEnabled = false; recBtn.alpha = 0.4f; nextBtn.visibility = View.GONE

        checklist = makeChecklist(); root.addView(checklist, FrameLayout.LayoutParams(-1, -1))
        setContentView(root)
    }

    private fun makeChecklist(): View {
        val c = Ui.column(this, 20).apply { setBackgroundColor(Ui.BG) }
        c.addView(Ui.text(this, "Before you film", 24f, bold = true))
        c.addView(Ui.text(this, "Tick each one. A room takes about 2 minutes, a flat about 3; up to 10 minutes per floor.", 14f, Ui.DIM))
        val boxes = Guide.checklist.map { s -> CheckBox(this).apply { text = s; textSize = 16f; setTextColor(Ui.TEXT); setPadding(8, 16, 8, 16) } }
        boxes.forEach { c.addView(it) }
        c.addView(Ui.wide(Ui.text(this, "While filming the app tells you what to do: three passes per room, and it warns you when you turn too fast, " +
            "aim at the floor, the picture blurs, or the light changes.", 14f, Ui.DIM), this, 12))
        val go = Ui.button(this, "Ready") {
            if (boxes.all { it.isChecked }) { checklist.visibility = View.GONE; passTitle.text = "Lock the camera"; say(Guide.LOCK, speak = true, warn = false) }
            else say("Tick all of them first", false)
        }
        c.addView(Ui.wide(go, this, 20))
        return ScrollView(this).apply { addView(c) }
    }

    private fun openCamera() {
        if (camReady || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val st = texture.surfaceTexture ?: return
        camReady = true
        try { rig.open(st) { runOnUiThread { info.text = "${rig.lens} · ${rig.size.width}x${rig.size.height} · 30 fps · stabilisation off" } } }
        catch (e: Exception) { camReady = false; say("Camera: ${e.message}", false) }
    }

    private fun toggleLock() {
        if (rig.recording) { say("The lock stays on while recording", false); return }
        rig.lock(!rig.locked)
        (lockBtn as android.widget.Button).text = if (rig.locked) "UNLOCK" else "LOCK"
        recBtn.isEnabled = rig.locked; recBtn.alpha = if (rig.locked) 1f else 0.4f
        if (rig.locked) { tip.text = "Locked (${rig.exposureInfo()}). Go to the entrance, then tap REC."; say("Locked. Go to the entrance, then tap record.", true, warn = false) }
        else tip.text = Guide.LOCK
    }

    private fun startRec() {
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        if (StatFs(dir.path).availableBytes < 2L shl 30) { say("Less than 2 GB free on the phone: free some space first", false); return }
        job = Jobs.newId(); file = File(dir, "$job.mp4")
        try { rig.start(file!!) } catch (e: Exception) { say("Could not start recording: ${e.message}", false); return }
        startedAt = SystemClock.elapsedRealtime(); pass = 0; room = 1
        recBtn.text = "■  STOP"; lockBtn.alpha = 0.4f; nextBtn.visibility = View.VISIBLE
        showPass(); listen(true); tickUi()
    }

    private fun showPass() {
        val p = Guide.passes[pass]
        passTitle.text = "Room $room · ${p.title}"
        tip.text = p.tip + "\n" + Guide.rules.joinToString("\n") { "· $it" }
        passLog.put(JSONObject().put("room", room).put("pass", p.title).put("t", (SystemClock.elapsedRealtime() - startedAt) / 1000.0))
        say(p.say, speak = true, warn = false, show = false)
    }

    private fun nextPass() {
        if (pass < Guide.passes.size - 1) pass++ else { room++; pass = 0 }
        showPass()
    }

    private fun tickUi() {
        if (!rig.recording) return
        val s = (SystemClock.elapsedRealtime() - startedAt) / 1000
        timer.text = "REC  %d:%02d   ·   turn %.0f°/s".format(s / 60, s % 60, coach.turnDegPerS)
        if (texture.isAvailable) {                                                   // the preview picture: blur / dark / lights
            texture.getBitmap(frame); frame.getPixels(px, 0, 72, 0, 0, 72, 128)
            for (i in px.indices) { val c = px[i]; luma[i] = ((c shr 16 and 255) * 299 + (c shr 8 and 255) * 587 + (c and 255) * 114) / 1000 }
            coach.frame(now(), Coach.mean(luma), Coach.sharpness(luma, 72, 128))?.let { show(it) }
        }
        if (SystemClock.elapsedRealtime() - lastCueAt > 2500) cue.visibility = View.GONE
        timer.postDelayed({ tickUi() }, 250)
    }

    private fun now() = SystemClock.elapsedRealtime()

    private fun listen(on: Boolean) {
        val sm = getSystemService(SensorManager::class.java)!!
        if (!on) { sm.unregisterListener(this); return }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        (sm.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER))?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (!rig.recording) return
        val v = e.values
        val c = when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> coach.gyro(now(), v[0], v[1], v[2])
            else -> coach.gravity(now(), v[0], v[1], v[2])
        }
        c?.let { show(it) }
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    private fun show(c: Coach.Cue) = say(c.text, speak = true, warn = true)

    private fun say(text: String, speak: Boolean, warn: Boolean = true, show: Boolean = true) {
        if (show) { cue.text = text; cue.setBackgroundColor(if (warn) 0xCCE5484D.toInt() else 0xCC2BB3A3.toInt()); cue.visibility = View.VISIBLE; lastCueAt = now() }
        if (speak) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue")
        if (warn && speak) vibrate()
    }

    private fun vibrate() {
        @Suppress("DEPRECATION") val v = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
        if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    private fun stopRec() {
        rig.stop(); listen(false)
        val f = file ?: return; val seconds = (SystemClock.elapsedRealtime() - startedAt) / 1000
        recBtn.text = "●  REC"; lockBtn.alpha = 1f; nextBtn.visibility = View.GONE; cue.visibility = View.GONE
        if (!f.exists() || f.length() < 100_000) { say("The recording is empty, try again", false); f.delete(); return }
        val name = EditText(this).apply { setText("Room ${java.text.SimpleDateFormat("d MMM HH:mm", Locale.getDefault()).format(java.util.Date())}"); setSelectAllOnFocus(true) }
        AlertDialog.Builder(this).setTitle("Name this recording (${seconds / 60} min ${seconds % 60} s)").setView(name).setCancelable(false)
            .setPositiveButton("Upload") { _, _ -> save(f, name.text.toString().ifBlank { "Room" }, seconds) }
            .setNegativeButton("Delete") { _, _ -> f.delete() }
            .show()
    }

    private fun save(f: File, name: String, seconds: Long) {
        val j = LocalJob(job, name, f.path, System.currentTimeMillis()).apply { total = f.length() }
        val meta = JSONObject().put("name", name).put("preset", "best").put("seconds", seconds).put("rooms", room)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL} (Android ${Build.VERSION.RELEASE})")
            .put("camera", JSONObject().put("lens", rig.lens).put("zoom", rig.zoom.toDouble()).put("width", rig.size.width).put("height", rig.size.height)
                .put("fps", 30).put("bitrate", rig.bitrate).put("locked", rig.locked).put("exposure", rig.exposureInfo()).put("stabilisation", "off"))
            .put("passes", passLog).put("cues", JSONObject(coach.counts.mapKeys { it.key.name }.mapValues { it.value }))
        j.meta.writeText(meta.toString(1))
        Store.add(this, j); UploadService.start(this); CheckJob.schedule(this)
        say("Saved. Uploading in the background; you can close the app.", true, warn = false)
        texture.postDelayed({ finish() }, 1500)
    }

    override fun onResume() { super.onResume(); if (texture.isAvailable) openCamera() }

    override fun onPause() {
        if (rig.recording) stopRec()                                    // leaving mid-recording: keep what was filmed
        rig.close(); camReady = false                                   // the camera is given back; LOCK again after returning
        (lockBtn as android.widget.Button).text = "LOCK"; recBtn.isEnabled = false; recBtn.alpha = 0.4f
        if (checklist.visibility != View.VISIBLE) tip.text = Guide.LOCK
        super.onPause()
    }

    override fun onDestroy() { listen(false); rig.quit(); tts?.shutdown(); super.onDestroy() }
}
