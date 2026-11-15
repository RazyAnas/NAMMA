package namma.core

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A made-up room and a walk through it, as the depth images and camera poses a phone would deliver: a 4 x 3 m room, 2.6 m
 * high, with a table in the middle; the camera walks a circle at eye height, first looking at the walls, then (second lap)
 * across the room. Used by the live-coverage screen's demo mode (no ARCore, no camera: to see the overlay and the numbers
 * work) and by CoverageTest, through exactly the same Coverage.addDepth call the real depth images take.
 */
class CoverageDemo(val w: Int = 96, val h: Int = 54, fovXDeg: Double = 70.0) {
    val fx = (w / 2.0 / Math.tan(Math.toRadians(fovXDeg / 2))).toFloat(); val fy = fx; val px = w / 2f; val py = h / 2f
    val depthMm = ShortArray(w * h)
    val pose = FloatArray(16)                       // camera to world, column-major, OpenGL axes (see Coverage.addDepth)
    val lapSeconds = 24.0

    /** The camera at time t (seconds): fills [pose] and [depthMm]. */
    fun step(t: Double) {
        val lap = t / lapSeconds; val a = 2 * Math.PI * lap; val inward = lap.toInt() % 2 == 1
        val ex = (1.1 * cos(a)).toFloat(); val ez = (0.8 * sin(a)).toFloat(); val ey = 1.4f
        val yaw = if (inward) a + Math.PI else a                                         // outward: at the wall it walks along
        val pitch = Math.toRadians(if (inward) -22.0 else 4.0 * sin(3 * a))
        val f = floatArrayOf((cos(pitch) * cos(yaw)).toFloat(), sin(pitch).toFloat(), (cos(pitch) * sin(yaw)).toFloat())
        val r = norm(floatArrayOf(-f[2], 0f, f[0])); val u = floatArrayOf(r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
        pose[0] = r[0]; pose[1] = r[1]; pose[2] = r[2]; pose[3] = 0f; pose[4] = u[0]; pose[5] = u[1]; pose[6] = u[2]; pose[7] = 0f
        pose[8] = -f[0]; pose[9] = -f[1]; pose[10] = -f[2]; pose[11] = 0f; pose[12] = ex; pose[13] = ey; pose[14] = ez; pose[15] = 1f
        for (v in 0 until h) for (c in 0 until w) {
            val xc = (c - px) / fx; val yc = -(v - py) / fy                               // the ray through this pixel, camera axes
            val dx = r[0] * xc + u[0] * yc + f[0]; val dy = r[1] * xc + u[1] * yc + f[1]; val dz = r[2] * xc + u[2] * yc + f[2]
            val s = hit(ex, ey, ez, dx, dy, dz)                                            // distance along the optical axis
            depthMm[v * w + c] = if (s.isFinite() && s < 60f) (s * 1000f).toInt().toShort() else 0
        }
    }

    /** Where the ray from (ox, oy, oz) along d first meets the room's inside or the table, as a multiple of d. */
    private fun hit(ox: Float, oy: Float, oz: Float, dx: Float, dy: Float, dz: Float): Float {
        var t = Float.MAX_VALUE
        fun wall(o: Float, d: Float, lo: Float, hi: Float) { if (d > 1e-6f) t = minOf(t, (hi - o) / d) else if (d < -1e-6f) t = minOf(t, (lo - o) / d) }
        wall(ox, dx, -2f, 2f); wall(oy, dy, 0f, 2.6f); wall(oz, dz, -1.5f, 1.5f)
        var t0 = 0f; var t1 = t                                                           // the table: a box, slab test
        fun slab(o: Float, d: Float, lo: Float, hi: Float): Boolean {
            if (kotlin.math.abs(d) < 1e-6f) return o in lo..hi
            val a = (lo - o) / d; val b = (hi - o) / d; t0 = maxOf(t0, minOf(a, b)); t1 = minOf(t1, maxOf(a, b)); return t0 <= t1
        }
        if (slab(ox, dx, -0.5f, 0.5f) && slab(oy, dy, 0f, 0.75f) && slab(oz, dz, -0.3f, 0.3f) && t0 > 0f) t = t0
        return t
    }

    private fun norm(v: FloatArray): FloatArray { val l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); return floatArrayOf(v[0] / l, v[1] / l, v[2] / l) }
}
