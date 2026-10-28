package namma.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CoachTest {
    @Test fun slowTurnIsFineFastTurnIsNot() {
        val c = Coach(); var t = 0L
        repeat(100) { t += 20; assertNull(c.gyro(t, 0f, Math.toRadians(25.0).toFloat(), 0f)) }      // 25 deg/s: a 3.6 s quarter turn
        val cues = (0 until 100).mapNotNull { t += 20; c.gyro(t, 0f, Math.toRadians(60.0).toFloat(), 0f) }
        assertEquals(listOf(Coach.Cue.TURN), cues)                                                   // once, then the cooldown
    }

    @Test fun aimFromGravity() {
        assertEquals(0.0, Coach.elevation(0f, 9.81f, 0f), 1e-6)                                     // upright, looking ahead
        assertEquals(-90.0, Coach.elevation(0f, 0f, 9.81f), 1e-6)                                   // flat, screen up: camera at the floor
        val c = Coach()
        assertNull(c.gravity(0, 0f, 9.6f, 2f))                                                      // ~12 deg down: fine
        assertEquals(Coach.Cue.FLOOR, c.gravity(10, 0f, 7f, 6.9f))                                  // ~45 deg down
        assertEquals(Coach.Cue.CEILING, c.gravity(20, 0f, 7f, -6.9f))
    }

    @Test fun blurAndDarkFromFrames() {
        val w = 64; val h = 48; val r = Random(1)
        val sharp = IntArray(w * h) { r.nextInt(40, 220) }
        val blurred = IntArray(w * h) { i -> val x = i % w; val y = i / w; if (x == 0 || y == 0 || x == w - 1 || y == h - 1) sharp[i] else
            (sharp[i] + sharp[i - 1] + sharp[i + 1] + sharp[i - w] + sharp[i + w]) / 5 }
        val blurred2 = IntArray(w * h) { i -> val x = i % w; val y = i / w; if (x == 0 || y == 0 || x == w - 1 || y == h - 1) blurred[i] else
            (blurred[i] + blurred[i - 1] + blurred[i + 1] + blurred[i - w] + blurred[i + w]) / 5 }
        val s0 = Coach.sharpness(sharp, w, h); val s2 = Coach.sharpness(blurred2, w, h)
        assertTrue(s2 < 0.2 * s0, "blur must cut the Laplacian variance ($s0 -> $s2)")
        val c = Coach(); var t = 0L
        repeat(10) { t += 200; assertNull(c.frame(t, 120.0, s0)) }
        val cues = (0 until 3).mapNotNull { t += 200; c.frame(t, 120.0, s2) }
        assertEquals(listOf(Coach.Cue.BLUR), cues)
        assertEquals(Coach.Cue.DARK, c.frame(t + 200, 20.0, s0))
    }

    @Test fun jobIdsAndKicks() {
        val id = Jobs.newId(0L, Random(3)); assertTrue(Regex("19700101-000000-[0-9a-f]{4}").matches(id), id)
        assertTrue(Jobs.needsKick(listOf("done", "uploaded"), 0, 0, 30 * 60_000L))
        assertTrue(!Jobs.needsKick(listOf("uploaded"), 1, 0, 30 * 60_000L))                        // a run is going
        assertTrue(!Jobs.needsKick(listOf("done", "failed"), 0, 0, 30 * 60_000L))
        assertTrue(!Jobs.needsKick(listOf("training"), 0, 25 * 60_000L, 30 * 60_000L))              // just kicked
        assertEquals("Ready", Status.parse("""{"stage":"done","link":"x"}""")!!.words)
    }
}
