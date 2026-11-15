package namma.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CoverageTest {
    /** A wall in the plane z = -2, x in -1..1, y in 0..2, sampled every 2 cm, seen from the camera at (cx, cy, cz). */
    private fun wall(c: Coverage, cx: Float, cy: Float, cz: Float, x0: Float = -1f, x1: Float = 1f) {
        var x = x0
        while (x < x1) { var y = 0f; while (y < 2f) { c.add(x, y, -2f, cx, cy, cz); y += 0.02f }; x += 0.02f }
    }

    @Test fun oneLookIsRedThreeDirectionsAreGreen() {
        val c = Coverage(0.04f)
        wall(c, 0f, 1.4f, 0f)
        var s = c.stats()
        assertTrue(s.seen > 2000 && s.good == 0 && s.percent == 0, "one position: nothing is covered well yet ($s)")
        assertTrue(s.weak > 0.5 * s.seen, "most of it has one direction only ($s)")          // the wall's far ends fall in other 30-degree bins
        wall(c, -3f, 1.4f, 0f); wall(c, 3f, 1.4f, 0f)                                       // the same wall from far left and far right
        s = c.stats()
        assertTrue(s.percent >= 90, "three directions: covered ($s)")
        val snap = c.snapshot()
        assertEquals(s.seen, snap.n); assertTrue(snap.level.all { it in 1..3 })
        assertTrue((0 until snap.n).all { abs(snap.xyz[3 * it + 2] + 2f) < 0.021f }, "voxel positions lie on the wall")
    }

    @Test fun aSingleHitIsNoiseAndIsNotShown() {
        val c = Coverage(); c.add(1f, 1f, 1f, 0f, 0f, 0f)
        assertEquals(0, c.stats().seen); assertEquals(0, c.snapshot().n); assertEquals(1, c.count)
        c.add(1.001f, 1f, 1f, 0f, 0f, 0f)
        assertEquals(1, c.stats().seen); assertEquals(1, c.stats().weak)
    }

    @Test fun directionBinsAre30DegreeHeadingsLowAndHigh() {
        val seen = HashSet<Int>()
        for (k in 0 until 12) { val a = Math.toRadians(k * 30.0 + 15.0); seen += Coverage.bin(cos(a).toFloat(), 0f, sin(a).toFloat()) }
        assertEquals((0 until 12).toSet(), seen)                                           // every heading its own bin, all in the low row
        assertEquals(12, Coverage.bin(1f, 1f, 0f) - Coverage.bin(1f, 0f, 0f))                 // from above: the high row
        assertEquals(Coverage.bin(1f, 0.2f, 0f), Coverage.bin(1f, 0f, 0f))                    // a little above eye level still counts as low
        assertTrue(Coverage.bin(-1f, 0f, 0f) in 0..11 && Coverage.bin(0f, 0f, 0f) in 0..11)
    }

    @Test fun depthImageLandsWhereTheCameraLooks() {
        val w = 160; val h = 90; val depth = ShortArray(w * h) { 2000 }                     // a flat wall 2 m ahead
        val id = FloatArray(16).also { it[0] = 1f; it[5] = 1f; it[10] = 1f; it[15] = 1f; it[13] = 1.4f }   // at eye height, looking down -z
        val c = Coverage(0.05f)
        val n = c.addDepth(depth, w, h, 120f, 120f, 80f, 45f, id, stride = 1); c.addDepth(depth, w, h, 120f, 120f, 80f, 45f, id, stride = 1)
        assertEquals(w * h, n)
        val s = c.snapshot(); assertTrue(s.n > 500)
        for (k in 0 until s.n) assertEquals(-2f, s.xyz[3 * k + 2], 0.026f)                    // 2 m in front = z -2
        val xs = (0 until s.n).map { s.xyz[3 * it] }; val ys = (0 until s.n).map { s.xyz[3 * it + 1] }
        assertEquals(-80f * 2 / 120, xs.min(), 0.06f); assertEquals(79f * 2 / 120, xs.max(), 0.06f)          // image left = -x
        assertEquals(1.4f + 45f * 2 / 120, ys.max(), 0.06f); assertEquals(1.4f - 44f * 2 / 120, ys.min(), 0.06f)   // image top = up
        // the camera turned 90 degrees to its left (now looking down -x): the wall is at x = -2
        val left = FloatArray(16).also { it[2] = -1f; it[5] = 1f; it[8] = 1f; it[15] = 1f }    // columns: right = (0,0,-1), up, back = (1,0,0)
        val c2 = Coverage(0.05f); repeat(2) { c2.addDepth(depth, w, h, 120f, 120f, 80f, 45f, left, stride = 1) }
        val s2 = c2.snapshot(); for (k in 0 until s2.n) assertEquals(-2f, s2.xyz[3 * k], 0.026f)
        // too near, too far and "no depth" pixels are ignored
        val bad = ShortArray(w * h) { i -> if (i % 3 == 0) 0 else if (i % 3 == 1) 100 else 9000 }
        assertEquals(0, Coverage().addDepth(bad, w, h, 120f, 120f, 80f, 45f, id, stride = 1))
        assertEquals(40 * 22, Coverage().addDepth(depth, w, h, 120f, 120f, 80f, 45f, id, stride = 4))   // every 4th pixel: 40 columns x 22 rows
    }

    @Test fun aFullMapDoublesItsVoxelsAndKeepsWhatItKnew() {
        val c = Coverage(0.04f, maxVoxels = 5000); val r = Random(7); var added = 0
        while (c.doublings == 0) {                                                          // a 4 x 4 m floor, twice per spot
            val x = r.nextFloat() * 4f; val z = r.nextFloat() * 4f
            c.add(x, 0f, z, 2f, 1.4f, 2f); c.add(x, 0f, z, 2f, 1.4f, 2f); added += 2
            assertTrue(added < 200_000, "the map never filled")
        }
        assertEquals(0.08f, c.voxel, 1e-6f); assertTrue(c.count < 5000 && c.count > 1000, "eight voxels became one: ${c.count}")
        val s = c.stats(); assertTrue(s.seen == c.count && s.voxelM == 0.08f)                  // every merged voxel still counts as seen
        val snap = c.snapshot(); assertTrue((0 until snap.n).all { snap.xyz[3 * it] in -0.01f..4.01f && abs(snap.xyz[3 * it + 1]) < 1e-4f })
        val before = c.count; wall(c, 0f, 1.4f, 0f); assertTrue(c.count > before)             // and it keeps taking new surface
        // a long walk never grows without bound: 60 m of corridor floor into a 5,000-voxel map
        val long = Coverage(0.04f, maxVoxels = 5000)
        for (i in 0 until 120_000) { val x = r.nextFloat() * 60f; val z = r.nextFloat() * 1.5f; long.add(x, 0f, z, x, 1.4f, z) }
        assertTrue(long.count <= 5000 && long.doublings >= 2 && long.voxel >= 0.16f, "count ${long.count}, voxel ${long.voxel}")
    }

    @Test fun tableGrowsAndFindsEveryVoxelAgain() {
        val c = Coverage(0.04f, maxVoxels = 1_000_000)
        for (x in 0 until 60) for (y in 0 until 30) for (z in 0 until 40) c.add(x * 0.04f + 0.02f, y * 0.04f + 0.02f, z * 0.04f + 0.02f, 0f, 0f, 0f)
        assertEquals(60 * 30 * 40, c.count); assertEquals(0, c.stats().seen)                  // 72,000 voxels, each seen once
        for (x in 0 until 60) for (y in 0 until 30) for (z in 0 until 40) c.add(x * 0.04f + 0.02f, y * 0.04f + 0.02f, z * 0.04f + 0.02f, 0f, 0f, 0f)
        assertEquals(60 * 30 * 40, c.count); assertEquals(60 * 30 * 40, c.stats().seen)        // the second pass found all of them
        c.add(-0.01f, -0.01f, -0.01f, 0f, 0f, 0f); assertEquals(60 * 30 * 40 + 1, c.count)     // negative coordinates are their own voxel
        c.clear(); assertEquals(0, c.count); assertEquals(0, c.snapshot().n)
    }

    @Test fun weakestPointsAtTheSurfaceThatNeedsAnotherLook() {
        val c = Coverage(0.04f)
        wall(c, 0f, 1.4f, 0f, x0 = -1f, x1 = 0f); wall(c, -3f, 1.4f, 0f, x0 = -1f, x1 = 0f); wall(c, 3f, 1.4f, 0f, x0 = -1f, x1 = 0f)   // left half: well covered
        assertNull(c.weakest(0f, 1.4f, 0f), "nothing weak yet")
        wall(c, 0.5f, 1.4f, 0f, x0 = 0.2f, x1 = 0.8f)                                         // right part: one look only
        val w = assertNotNull(c.weakest(0f, 1.4f, 0f))
        assertTrue(w.x in 0.2f..0.8f && abs(w.z + 2f) < 0.05f && w.voxels >= 25, "weak spot at ${w.x}, ${w.y}, ${w.z}")
        assertNull(c.weakest(0f, 1.4f, 20f), "too far away to send the user there")
    }

    @Test fun topDownMapShowsTheBestLevelPerSquare() {
        val c = Coverage(0.04f); assertNull(c.topDown())
        wall(c, 0f, 1.4f, 0f); wall(c, -3f, 1.4f, 0f, x0 = -1f, x1 = 0f); wall(c, 3f, 1.4f, 0f, x0 = -1f, x1 = 0f)
        val m = assertNotNull(c.topDown(0.1f))
        assertTrue(m.w in 19..21 && m.h == 1 && abs(m.x0 + 1f) < 0.05f && abs(m.z0 + 2f) < 0.05f, "a 2 m wall seen from above: ${m.w} x ${m.h}")
        assertTrue(m.level.take(8).all { it.toInt() == 3 } && m.level.takeLast(5).all { it.toInt() in 1..2 }, "left end green, right end not yet")
    }
}
