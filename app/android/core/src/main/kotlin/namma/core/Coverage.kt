package namma.core

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Live scan coverage (16 Nov): what the camera has already seen well, what it has only glanced at, and what it has not seen.
 * Depth frames with their camera pose are fused into a sparse map of small cubes (voxels) on the surfaces of the room; each
 * voxel remembers from which directions it was seen (24 bins: 12 headings of 30 degrees x low / high). A photoreal model
 * needs every surface from several directions, so:
 *   level 0  seen once only (depth noise): not shown
 *   level 1  one direction    (red: come back to it)
 *   level 2  two directions   (yellow)
 *   level 3  three or more    (green: enough)
 * Unpainted surface has not been seen at all. Same rule as the web recorder's coverage.js, plus what a whole house needs:
 * the map lives in one open-addressing hash table of plain arrays (no object per voxel), and when it reaches [maxVoxels]
 * the voxel size doubles and the map is rebuilt at the coarser size, so memory stays bounded however long the walk is.
 * Pure Kotlin, no Android: CoverageTest feeds it synthetic rooms. Not thread-safe: call it from one thread (the AR frame
 * thread) and hand [snapshot]s to the renderer.
 */
class Coverage(voxel: Float = 0.04f, private val maxVoxels: Int = 120_000) {
    var voxel = voxel; private set
    var count = 0; private set                      // voxels in the map
    var doublings = 0; private set
    var version = 0; private set                    // changes whenever the map does: the renderer re-uploads only then

    private var cap = 1 shl 15                      // slots, a power of two
    private var keys = LongArray(cap) { EMPTY }
    private var hits = IntArray(cap)
    private var dirs = IntArray(cap)
    private var sx = FloatArray(cap); private var sy = FloatArray(cap); private var sz = FloatArray(cap)   // summed positions

    data class Stats(val seen: Int, val good: Int, val weak: Int, val percent: Int, val voxelM: Float)
    class Snapshot(val n: Int, val xyz: FloatArray, val level: ByteArray, val voxelM: Float, val version: Int)
    class Map2d(val x0: Float, val z0: Float, val cell: Float, val w: Int, val h: Int, val level: ByteArray)
    class Spot(val x: Float, val y: Float, val z: Float, val voxels: Int)

    /** One surface point (world, metres) seen from the camera at (cx, cy, cz). */
    fun add(x: Float, y: Float, z: Float, cx: Float, cy: Float, cz: Float) {
        val i = slot(key(x, y, z)); hits[i]++; dirs[i] = dirs[i] or (1 shl bin(cx - x, cy - y, cz - z))
        if (hits[i] < MEAN_HITS) { sx[i] += x; sy[i] += y; sz[i] += z }        // the mean settles; later hits only count
        version++
        if (count >= maxVoxels) coarsen()
    }

    /** n points as x, y, z triples. */
    fun addPoints(p: FloatArray, n: Int, cx: Float, cy: Float, cz: Float) { for (k in 0 until n) add(p[3 * k], p[3 * k + 1], p[3 * k + 2], cx, cy, cz) }

    /**
     * One depth image. [depthMm] = w x h distances along the optical axis in millimetres, 0 = no depth (ARCore's 16-bit
     * depth image); fx, fy, px, py = the intrinsics of that image in pixels; [camToWorld] = 4x4 column-major pose of the camera
     * in the world with OpenGL axes (+x right, +y up, -z where the camera looks) lined up with the image (image right = +x,
     * image down = -y), as ARCore's Camera.getPose() is for its camera image. Every [stride]-th pixel between minM and maxM.
     * Returns the number of points fused.
     */
    fun addDepth(depthMm: ShortArray, w: Int, h: Int, fx: Float, fy: Float, px: Float, py: Float, camToWorld: FloatArray,
                 stride: Int = 2, minM: Float = 0.25f, maxM: Float = 4.5f): Int {
        val m = camToWorld; val cx = m[12]; val cy = m[13]; val cz = m[14]; var n = 0
        var v = stride / 2
        while (v < h) {
            var u = stride / 2
            while (u < w) {
                val d = (depthMm[v * w + u].toInt() and 0xFFFF) * 0.001f
                if (d in minM..maxM) {
                    val xc = (u - px) * d / fx; val yc = -(v - py) * d / fy; val zc = -d
                    add(m[0] * xc + m[4] * yc + m[8] * zc + cx, m[1] * xc + m[5] * yc + m[9] * zc + cy, m[2] * xc + m[6] * yc + m[10] * zc + cz, cx, cy, cz); n++
                }
                u += stride
            }
            v += stride
        }
        return n
    }

    /** Share of the confirmed surface (seen at least twice) that is seen from three or more directions. */
    fun stats(): Stats {
        var seen = 0; var good = 0; var weak = 0
        for (i in 0 until cap) if (keys[i] != EMPTY) when (level(i)) { 0 -> {}; 3 -> { seen++; good++ }; 1 -> { seen++; weak++ }; else -> seen++ }
        return Stats(seen, good, weak, if (seen == 0) 0 else Math.round(100f * good / seen), voxel)
    }

    /** The confirmed voxels for drawing: mean position of each and its level (1..3). A copy: safe to hand to another thread. */
    fun snapshot(): Snapshot {
        var n = 0
        for (i in 0 until cap) if (keys[i] != EMPTY && level(i) > 0) n++
        val xyz = FloatArray(3 * n); val lv = ByteArray(n); var k = 0
        for (i in 0 until cap) {
            if (keys[i] == EMPTY) continue
            val l = level(i); if (l == 0) continue
            val c = minOf(hits[i], MEAN_HITS - 1).toFloat()
            xyz[3 * k] = sx[i] / c; xyz[3 * k + 1] = sy[i] / c; xyz[3 * k + 2] = sz[i] / c; lv[k] = l.toByte(); k++
        }
        return Snapshot(n, xyz, lv, voxel, version)
    }

    /**
     * Where to go back to: the half-metre block within [radius] of the camera that holds the most red (one-direction)
     * surface, as its centre, or null if no block has [minVoxels] of them. For an arrow and "this corner needs another look".
     */
    fun weakest(cx: Float, cy: Float, cz: Float, radius: Float = 4f, minVoxels: Int = 25): Spot? {
        val blocks = HashMap<Long, FloatArray>(); val r2 = radius * radius                    // block -> n, sum x, y, z
        for (i in 0 until cap) {
            if (keys[i] == EMPTY || level(i) != 1) continue
            val c = minOf(hits[i], MEAN_HITS - 1).toFloat(); val x = sx[i] / c; val y = sy[i] / c; val z = sz[i] / c
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz) > r2) continue
            val b = blocks.getOrPut(pack(floor(x / 0.5f).toInt(), floor(y / 0.5f).toInt(), floor(z / 0.5f).toInt())) { FloatArray(4) }
            b[0] += 1f; b[1] += x; b[2] += y; b[3] += z
        }
        val best = blocks.values.maxByOrNull { it[0] } ?: return null
        return if (best[0] < minVoxels) null else Spot(best[1] / best[0], best[2] / best[0], best[3] / best[0], best[0].toInt())
    }

    /** Top-down map for a minimap: per [cell]-sized square of floor the best level of any voxel above it (0 = nothing seen). */
    fun topDown(cell: Float = 0.1f): Map2d? {
        val s = snapshot(); if (s.n == 0) return null
        var x0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE; var z0 = Float.MAX_VALUE; var z1 = -Float.MAX_VALUE
        for (k in 0 until s.n) { val x = s.xyz[3 * k]; val z = s.xyz[3 * k + 2]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z }
        val w = ((x1 - x0) / cell).toInt() + 1; val h = ((z1 - z0) / cell).toInt() + 1
        if (w.toLong() * h > 4_000_000L) return null                                       // an outlier far away: no map this time
        val lv = ByteArray(w * h)
        for (k in 0 until s.n) {
            val i = ((s.xyz[3 * k + 2] - z0) / cell).toInt() * w + ((s.xyz[3 * k] - x0) / cell).toInt()
            if (s.level[k] > lv[i]) lv[i] = s.level[k]
        }
        return Map2d(x0, z0, cell, w, h, lv)
    }

    fun clear() { keys.fill(EMPTY); hits.fill(0); dirs.fill(0); sx.fill(0f); sy.fill(0f); sz.fill(0f); count = 0; version++ }

    // ---- the table
    private fun level(i: Int): Int = if (hits[i] < 2) 0 else minOf(3, Integer.bitCount(dirs[i]))

    private fun key(x: Float, y: Float, z: Float) = pack(floor(x / voxel).toInt(), floor(y / voxel).toInt(), floor(z / voxel).toInt())

    /** The slot of [k], taking a free one (and growing the table past 60 % full) if it is new. */
    private fun slot(k: Long): Int {
        var i = hash(k) and (cap - 1)
        while (true) {
            if (keys[i] == k) return i
            if (keys[i] == EMPTY) break
            i = (i + 1) and (cap - 1)
        }
        if ((count + 1) * 10 > cap * 6) { grow(); return slot(k) }
        keys[i] = k; count++; return i
    }

    private fun grow() {
        val ok = keys; val oh = hits; val od = dirs; val ox = sx; val oy = sy; val oz = sz
        cap *= 2; alloc()
        for (j in ok.indices) if (ok[j] != EMPTY) { val i = slot(ok[j]); hits[i] = oh[j]; dirs[i] = od[j]; sx[i] = ox[j]; sy[i] = oy[j]; sz[i] = oz[j] }
    }

    /** The map is full: twice the voxel size, eight old voxels fall into one (hits add up, directions combine). */
    private fun coarsen() {
        val ok = keys; val oh = hits; val od = dirs; val ox = sx; val oy = sy; val oz = sz
        voxel *= 2f; doublings++; alloc()
        for (j in ok.indices) {
            if (ok[j] == EMPTY) continue
            val c = minOf(oh[j], MEAN_HITS - 1).toFloat(); val x = ox[j] / c; val y = oy[j] / c; val z = oz[j] / c
            val i = slot(key(x, y, z)); val had = minOf(hits[i], MEAN_HITS - 1); val add = minOf(oh[j], MEAN_HITS - 1 - had)
            hits[i] += oh[j]; dirs[i] = dirs[i] or od[j]
            if (add > 0) { sx[i] += x * add; sy[i] += y * add; sz[i] += z * add }             // keep sum / min(hits, MEAN_HITS - 1) = mean
        }
        version++
    }

    private fun alloc() { keys = LongArray(cap) { EMPTY }; hits = IntArray(cap); dirs = IntArray(cap); sx = FloatArray(cap); sy = FloatArray(cap); sz = FloatArray(cap); count = 0 }

    companion object {
        private const val EMPTY = Long.MIN_VALUE
        private const val MEAN_HITS = 64                    // positions are averaged over the first 63 hits
        private const val OFF = 1 shl 20                    // voxel coordinates -2^20 .. 2^20 (40 km at 4 cm)

        fun pack(x: Int, y: Int, z: Int): Long = ((x + OFF).toLong() shl 42) or ((y + OFF).toLong() shl 21) or (z + OFF).toLong()

        private fun hash(k: Long): Int { var h = k * -0x61c8864680b583ebL; h = h xor (h ushr 32); return h.toInt() }

        /** Direction bin (0..23) of the camera as seen from the surface point: 12 headings of 30 degrees, low or high. */
        fun bin(dx: Float, dy: Float, dz: Float): Int {
            val heading = (floor((atan2(dz.toDouble(), dx.toDouble()) + PI) / (2 * PI) * 12).toInt()).mod(12)
            val high = if (dy > 0.35f * sqrt(dx * dx + dy * dy + dz * dz)) 1 else 0
            return heading + 12 * high
        }
    }
}
