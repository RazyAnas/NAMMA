package namma.capture

import android.opengl.GLES11Ext
import android.opengl.GLES20
import namma.core.Coverage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** The two things the live-coverage screen draws (OpenGL ES 2, GL thread only): the camera picture as the background
 *  (ARCore's external texture) and the coverage map on top of it, one soft square per surface voxel in its level's colour
 *  (red = seen from one direction, yellow = two, green = three or more). Surfaces without paint have not been seen. */
class CoverageGl {
    var cameraTexture = 0; private set
    private var bgProg = 0; private var ptProg = 0
    private val quad: FloatBuffer = buf(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))        // NDC, triangle strip
    val quadNdc: FloatBuffer get() = quad
    val uv: FloatBuffer = buf(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))                         // ARCore rewrites these for the screen
    private var vbo = 0; private var points = 0; private var shownVersion = -1; private var voxelM = 0.04f

    fun create() {
        val t = IntArray(1); GLES20.glGenTextures(1, t, 0); cameraTexture = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        for (p in intArrayOf(GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_TEXTURE_WRAP_T)) GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, p, GLES20.GL_CLAMP_TO_EDGE)
        for (p in intArrayOf(GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_TEXTURE_MAG_FILTER)) GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, p, GLES20.GL_LINEAR)
        bgProg = program("attribute vec2 p; attribute vec2 t; varying vec2 v; void main() { gl_Position = vec4(p, 0.0, 1.0); v = t; }",
            "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 v; uniform samplerExternalOES s; void main() { gl_FragColor = texture2D(s, v); }")
        // point size: the voxel's size on screen (focal length in pixels x voxel / distance), so near surface is painted solid
        ptProg = program("attribute vec3 p; attribute vec4 c; uniform mat4 vp; uniform float px; varying vec4 vc;\n" +
            "void main() { gl_Position = vp * vec4(p, 1.0); gl_PointSize = clamp(px / max(gl_Position.w, 0.05), 2.0, 220.0); vc = c; }",
            "precision mediump float; varying vec4 vc; void main() { vec2 d = abs(gl_PointCoord - 0.5); float e = max(d.x, d.y);\n" +
            "  gl_FragColor = vec4(vc.rgb, vc.a * (e > 0.42 ? 2.2 : 1.0)); }")                       // a stronger rim: the grid reads as a mesh
        val b = IntArray(1); GLES20.glGenBuffers(1, b, 0); vbo = b[0]; shownVersion = -1; points = 0
    }

    /** The camera picture over the whole target. texUv = which part of the camera texture (default: the screen's crop). */
    fun background(texUv: FloatBuffer = uv) {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST); GLES20.glDepthMask(false); GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(bgProg); GLES20.glActiveTexture(GLES20.GL_TEXTURE0); GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(bgProg, "s"), 0)
        val p = GLES20.glGetAttribLocation(bgProg, "p"); val t = GLES20.glGetAttribLocation(bgProg, "t")
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        quad.position(0); GLES20.glVertexAttribPointer(p, 2, GLES20.GL_FLOAT, false, 0, quad); GLES20.glEnableVertexAttribArray(p)
        texUv.position(0); GLES20.glVertexAttribPointer(t, 2, GLES20.GL_FLOAT, false, 0, texUv); GLES20.glEnableVertexAttribArray(t)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(p); GLES20.glDisableVertexAttribArray(t)
    }

    /** A new picture of the map (only uploaded when it changed). */
    fun upload(s: Coverage.Snapshot) {
        if (s.version == shownVersion) return
        val data = ByteBuffer.allocateDirect(s.n * 16).order(ByteOrder.nativeOrder())               // x y z (floats) + rgba (bytes)
        for (k in 0 until s.n) {
            data.putFloat(s.xyz[3 * k]); data.putFloat(s.xyz[3 * k + 1]); data.putFloat(s.xyz[3 * k + 2])
            val c = COLOURS[s.level[k].toInt().coerceIn(1, 3)]; data.put(c[0]); data.put(c[1]); data.put(c[2]); data.put(c[3])
        }
        data.position(0); GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo); GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, s.n * 16, data, GLES20.GL_DYNAMIC_DRAW)
        points = s.n; shownVersion = s.version; voxelM = s.voxelM
    }

    /** The map over the picture. viewProj = projection x view (column-major); focalPx = the projection's focal length in pixels.
     *  Two passes, so every pixel is tinted once, by the nearest voxel only: first the depths, then the colours where the depth
     *  matches. (Without it every voxel behind a pixel adds its tint: a real room turned solid red, 16 Nov on the phone.) */
    fun coverage(viewProj: FloatArray, focalPx: Float) {
        if (points == 0) return
        GLES20.glEnable(GLES20.GL_DEPTH_TEST); GLES20.glDepthMask(true); GLES20.glClearDepthf(1f); GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glEnable(GLES20.GL_BLEND); GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(ptProg); GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(ptProg, "vp"), 1, false, viewProj, 0)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(ptProg, "px"), focalPx * voxelM * 1.05f)
        val p = GLES20.glGetAttribLocation(ptProg, "p"); val c = GLES20.glGetAttribLocation(ptProg, "c")
        GLES20.glVertexAttribPointer(p, 3, GLES20.GL_FLOAT, false, 16, 0); GLES20.glEnableVertexAttribArray(p)
        GLES20.glVertexAttribPointer(c, 4, GLES20.GL_UNSIGNED_BYTE, true, 16, 12); GLES20.glEnableVertexAttribArray(c)
        GLES20.glColorMask(false, false, false, false); GLES20.glDepthFunc(GLES20.GL_LESS); GLES20.glDrawArrays(GLES20.GL_POINTS, 0, points)
        GLES20.glColorMask(true, true, true, true); GLES20.glDepthMask(false); GLES20.glDepthFunc(GLES20.GL_LEQUAL); GLES20.glDrawArrays(GLES20.GL_POINTS, 0, points)
        GLES20.glDepthMask(true); GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisableVertexAttribArray(p); GLES20.glDisableVertexAttribArray(c); GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0); GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun program(vs: String, fs: String): Int {
        fun sh(type: Int, src: String): Int {
            val s = GLES20.glCreateShader(type); GLES20.glShaderSource(s, src); GLES20.glCompileShader(s)
            val ok = IntArray(1); GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
            check(ok[0] != 0) { "shader: " + GLES20.glGetShaderInfoLog(s) }
            return s
        }
        val p = GLES20.glCreateProgram(); GLES20.glAttachShader(p, sh(GLES20.GL_VERTEX_SHADER, vs)); GLES20.glAttachShader(p, sh(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(p); val ok = IntArray(1); GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "program: " + GLES20.glGetProgramInfoLog(p) }
        return p
    }

    companion object {
        fun buf(a: FloatArray): FloatBuffer = ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }
        private fun rgba(r: Int, g: Int, b: Int, a: Int) = byteArrayOf(r.toByte(), g.toByte(), b.toByte(), a.toByte())
        /** index = level: 1 red (one direction), 2 yellow (two), 3 green (three or more) */
        val COLOURS = arrayOf(rgba(0, 0, 0, 0), rgba(229, 72, 77, 85), rgba(245, 197, 66, 80), rgba(43, 179, 110, 85))   // the picture stays visible
        val COLOUR_INT = intArrayOf(0, 0xFFE5484D.toInt(), 0xFFF5C542.toInt(), 0xFF2BB36E.toInt())
    }
}
