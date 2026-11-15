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
    private var filmProg = 0; private var filmTex = 0; private var filmW = 0; private var filmH = 0; private var filmBuf: ByteBuffer? = null
    private var depthTex = 0; private var depthBuf: ByteBuffer? = null; private var filmLit = false
    /** Where the film's picture (the depth image's pixels) sits on the screen quad: the whole camera image by default. */
    val filmUv: FloatBuffer = buf(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))

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
        // the film: a small picture (one texel per depth pixel) stretched smoothly over the screen. Colours are stored already
        // multiplied by their opacity, so the smooth stretch does not darken the film's border. The border itself glows: where
        // film meets no film between neighbouring texels, a bright cyan line is added (the edge of what has been scanned).
        val vs = "attribute vec2 p; attribute vec2 t; varying vec2 v; void main() { gl_Position = vec4(p, 0.0, 1.0); v = t; }"
        val rim = "  vec2 h = px * 0.45;\n" +                                              // half a texel each way: a thin line
            "  float eL = smoothstep(0.1, 0.2, texture2D(s, v - vec2(h.x, 0.0)).a), eR = smoothstep(0.1, 0.2, texture2D(s, v + vec2(h.x, 0.0)).a);\n" +
            "  float eD = smoothstep(0.1, 0.2, texture2D(s, v - vec2(0.0, h.y)).a), eU = smoothstep(0.1, 0.2, texture2D(s, v + vec2(0.0, h.y)).a);\n" +
            "  float rim = clamp(abs(eR - eL) + abs(eU - eD), 0.0, 1.0) * 0.9;\n"         // scanned / not scanned only: no line where the film just gets thicker
        // The film as a surface (owner, 16 Nov: "it looks 2D, not things wrapped in plastic"). For every screen pixel the
        // depth image (smoothly interpolated: 4 texels) gives the 3D point under it; its neighbours give the surface's
        // direction there. That surface is lit like glossy plastic: a light fixed in the room (so a wall, a table top and a
        // cushion each keep their own brightness as the phone moves), a sharp highlight, a sheen where the surface turns away
        // from the camera, and a fine 10 cm grid drawn in the room's own coordinates, which bends over every object and shows
        // its shape. Where depth jumps from one object to the next the surface is steep and dark: each thing gets an outline.
        val lit = "#extension GL_OES_standard_derivatives : enable\nprecision highp float; varying vec2 v;\n" +
            "uniform sampler2D s; uniform sampler2D dz; uniform vec2 px; uniform vec2 size; uniform vec4 k; uniform mat3 rot; uniform vec3 eye; uniform vec3 lightCam;\n" +
            "float mm(vec2 uv) { vec4 c = texture2D(dz, uv); return (c.r * 255.0 + c.a * 65280.0) * 0.001; }\n" +         // metres; 0 = no depth
            "float depthAt(vec2 t) { vec2 f = fract(t - 0.5), b = (floor(t - 0.5) + 0.5) * px;\n" +
            "  float a = mm(b), c = mm(b + vec2(px.x, 0.0)), d = mm(b + vec2(0.0, px.y)), e = mm(b + px), m = max(max(a, c), max(d, e));\n" +
            "  if (a <= 0.0) a = m; if (c <= 0.0) c = m; if (d <= 0.0) d = m; if (e <= 0.0) e = m;\n" +                    // a missing texel takes a neighbour's depth
            "  return mix(mix(a, c, f.x), mix(d, e, f.x), f.y); }\n" +
            "vec3 cam(vec2 t, float d) { return vec3((t.x - k.z) * d / k.x, -(t.y - k.w) * d / k.y, -d); }\n" +            // camera axes: x right, y up, z back
            "void main() { vec4 c = texture2D(s, v);\n" + rim +
            "  vec3 col = c.rgb; float a = c.a; vec2 t = v * size; float d = depthAt(t);\n" +
            "  if (a > 0.004 && d > 0.0) {\n" +
            "    vec3 P = cam(t, d);\n" +
            "    vec3 dx = cam(t + vec2(1.0, 0.0), depthAt(t + vec2(1.0, 0.0))) - cam(t - vec2(1.0, 0.0), depthAt(t - vec2(1.0, 0.0)));\n" +
            "    vec3 dy = cam(t + vec2(0.0, 1.0), depthAt(t + vec2(0.0, 1.0))) - cam(t - vec2(0.0, 1.0), depthAt(t - vec2(0.0, 1.0)));\n" +
            "    vec3 n = normalize(cross(dx, dy)); vec3 vd = normalize(-P); if (dot(n, vd) < 0.0) n = -n;\n" +
            "    float facing = dot(n, vd); vec3 nw = rot * n;\n" +
            "    float diff = 0.34 + 0.46 * max(dot(n, lightCam), 0.0) + 0.20 * (0.5 + 0.5 * nw.y);\n" +                    // the room light + a little from the sky
            "    float spec = pow(max(dot(n, normalize(vd + lightCam)), 0.0), 40.0);\n" +
            "    float sheen = pow(1.0 - facing, 3.0);\n" +
            "    float crease = smoothstep(0.30, 0.08, facing);\n" +                                                        // steep = the gap between two objects
            "    vec3 g = (rot * P + eye) * 10.0; vec3 fw = max(fwidth(g), vec3(1e-5)); float top = max(max(fw.x, fw.y), fw.z);\n" +
            "    vec3 ln = abs(fract(g - 0.5) - 0.5) / fw; ln += step(fw, vec3(0.3 * top)) * 100.0;\n" +                      // an axis the surface runs along draws no lines
            "    float line = clamp(1.3 - min(min(ln.x, ln.y), ln.z), 0.0, 1.0) * (1.0 - smoothstep(0.2, 0.5, top));\n" +      // too far to tell lines apart: none
            "    col = c.rgb * diff * (1.0 - 0.55 * crease) + a * (vec3(spec * 0.8) + vec3(0.55, 0.9, 1.0) * (sheen * 0.35 + line * 0.5));\n" +
            "  }\n" +
            "  gl_FragColor = vec4(col * (1.0 - 0.6 * rim) + rim * vec3(0.35, 0.95, 1.0), max(a, rim)); }"
        // a phone whose graphics chip cannot do that: the flat film (colour by level + the edge)
        val flat = "precision mediump float; varying vec2 v; uniform sampler2D s; uniform vec2 px;\n" +
            "void main() { vec4 c = texture2D(s, v);\n" + rim + "  gl_FragColor = vec4(c.rgb * (1.0 - 0.6 * rim) + rim * vec3(0.35, 0.95, 1.0), max(c.a, rim)); }"
        filmLit = true
        filmProg = try { program(vs, lit) } catch (e: Exception) { android.util.Log.w("NammaAr", "lit film shader not available: ${e.message}"); filmLit = false; program(vs, flat) }
        val dt = IntArray(1); GLES20.glGenTextures(1, dt, 0); depthTex = dt[0]; GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
        for (q in intArrayOf(GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_TEXTURE_WRAP_T)) GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, q, GLES20.GL_CLAMP_TO_EDGE)
        for (q in intArrayOf(GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_TEXTURE_MAG_FILTER)) GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, q, GLES20.GL_NEAREST)   // packed 16-bit: the shader interpolates
        val ft = IntArray(1); GLES20.glGenTextures(1, ft, 0); filmTex = ft[0]; filmW = 0; filmH = 0
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, filmTex)
        for (q in intArrayOf(GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_TEXTURE_WRAP_T)) GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, q, GLES20.GL_CLAMP_TO_EDGE)
        for (q in intArrayOf(GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_TEXTURE_MAG_FILTER)) GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, q, GLES20.GL_LINEAR)
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

    /**
     * The film for this frame: per depth pixel the coverage level (Coverage.levels; 0 = clear) and the depth itself
     * (millimetres, as ARCore's 16-bit depth image). The level gives the film's colour and thickness, the depth its shape.
     */
    fun film(level: ByteArray, depthMm: ShortArray, w: Int, h: Int) {
        val b = filmBuf?.takeIf { it.capacity() == w * h * 4 } ?: ByteBuffer.allocateDirect(w * h * 4).also { filmBuf = it }
        b.position(0)
        for (i in 0 until w * h) {
            val l = level[i].toInt()
            if (l <= 0) { b.putInt(0); continue }
            val c = FILM[minOf(l, 3)]; val al = c[3]                                        // colour already multiplied by its opacity
            b.put((c[0] * al * 255f).toInt().toByte()); b.put((c[1] * al * 255f).toInt().toByte()); b.put((c[2] * al * 255f).toInt().toByte()); b.put((al * 255f).toInt().toByte())
        }
        val fresh = w != filmW || h != filmH
        b.position(0); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, filmTex)
        if (fresh) GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, b)
        else GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, b)
        if (filmLit) {                                                                      // depth: low byte in luminance, high byte in alpha
            val d = depthBuf?.takeIf { it.capacity() == w * h * 2 } ?: ByteBuffer.allocateDirect(w * h * 2).order(ByteOrder.LITTLE_ENDIAN).also { depthBuf = it }
            d.position(0); d.asShortBuffer().put(depthMm, 0, w * h); d.position(0)
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
            if (fresh) GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE_ALPHA, w, h, 0, GLES20.GL_LUMINANCE_ALPHA, GLES20.GL_UNSIGNED_BYTE, d)
            else GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, w, h, GLES20.GL_LUMINANCE_ALPHA, GLES20.GL_UNSIGNED_BYTE, d)
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 4)
        }
        filmW = w; filmH = h
    }

    /**
     * The film over the picture. k = the depth image's fx, fy, cx, cy in its own pixels; camToWorld = the camera pose (4 x 4,
     * column-major), which turns the surface into room coordinates for the light and the grid.
     */
    fun drawFilm(k: FloatArray, camToWorld: FloatArray) {
        if (filmW == 0) return
        GLES20.glDisable(GLES20.GL_DEPTH_TEST); GLES20.glEnable(GLES20.GL_BLEND); GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(filmProg); GLES20.glActiveTexture(GLES20.GL_TEXTURE0); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, filmTex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(filmProg, "s"), 0); GLES20.glUniform2f(GLES20.glGetUniformLocation(filmProg, "px"), 1f / filmW, 1f / filmH)
        if (filmLit) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex); GLES20.glUniform1i(GLES20.glGetUniformLocation(filmProg, "dz"), 1)
            GLES20.glUniform2f(GLES20.glGetUniformLocation(filmProg, "size"), filmW.toFloat(), filmH.toFloat())
            GLES20.glUniform4f(GLES20.glGetUniformLocation(filmProg, "k"), k[0], k[1], k[2], k[3])
            val m = camToWorld; val r = floatArrayOf(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10])
            GLES20.glUniformMatrix3fv(GLES20.glGetUniformLocation(filmProg, "rot"), 1, false, r, 0)
            GLES20.glUniform3f(GLES20.glGetUniformLocation(filmProg, "eye"), m[12], m[13], m[14])
            // the light, fixed in the room (from above, a little to one side), turned into camera axes: rot^T x L
            GLES20.glUniform3f(GLES20.glGetUniformLocation(filmProg, "lightCam"), m[0] * LIGHT[0] + m[1] * LIGHT[1] + m[2] * LIGHT[2],
                m[4] * LIGHT[0] + m[5] * LIGHT[1] + m[6] * LIGHT[2], m[8] * LIGHT[0] + m[9] * LIGHT[1] + m[10] * LIGHT[2])
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        }
        val p = GLES20.glGetAttribLocation(filmProg, "p"); val t = GLES20.glGetAttribLocation(filmProg, "t")
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        quad.position(0); GLES20.glVertexAttribPointer(p, 2, GLES20.GL_FLOAT, false, 0, quad); GLES20.glEnableVertexAttribArray(p)
        filmUv.position(0); GLES20.glVertexAttribPointer(t, 2, GLES20.GL_FLOAT, false, 0, filmUv); GLES20.glEnableVertexAttribArray(t)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(p); GLES20.glDisableVertexAttribArray(t); GLES20.glDisable(GLES20.GL_BLEND)
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
        val COLOURS = arrayOf(rgba(0, 0, 0, 0), rgba(229, 72, 77, 85), rgba(245, 197, 66, 80), rgba(60, 140, 255, 85))   // the picture stays visible
        /** Legend and top-down map, by level: one direction (thin, violet), two, three or more (covered: deep blue). */
        val COLOUR_INT = intArrayOf(0, 0xFFB49BF2.toInt(), 0xFF6FA2FF.toInt(), 0xFF2A7BFF.toInt())
        /** The film by level, r g b + opacity: one family of blue, like one sheet of plastic that gets thicker as a surface is
         *  seen from more directions (the red / amber / blue patchwork of the first phone test read as noise). */
        val FILM = arrayOf(floatArrayOf(0f, 0f, 0f, 0f), floatArrayOf(0.66f, 0.56f, 0.98f, 0.40f), floatArrayOf(0.40f, 0.62f, 1.0f, 0.54f), floatArrayOf(0.14f, 0.50f, 1.0f, 0.68f))
        /** The light on the film, in room axes (y up), unit length. */
        val LIGHT = floatArrayOf(0.35f, 0.89f, 0.29f)
    }
}
