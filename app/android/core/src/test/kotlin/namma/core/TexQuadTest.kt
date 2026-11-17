package namma.core

import kotlin.test.Test
import kotlin.test.assertContentEquals

class TexQuadTest {
    @Test fun `a 4 to 3 image on a 16 to 9 texture becomes the whole texture`() {
        // the first scan: image corners in texture coordinates, v outside 0..1 by a sixth, v flipped
        val uv = floatArrayOf(0f, 7 / 6f, 1f, 7 / 6f, 0f, -1 / 6f, 1f, -1 / 6f)
        TexQuad.whole(uv); assertContentEquals(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f), uv)
    }

    @Test fun `turned and mirrored corners keep their order`() {
        val uv = floatArrayOf(1.1f, 0.9f, 1.1f, 0.1f, -0.1f, 0.9f, -0.1f, 0.1f)          // a quarter turn
        TexQuad.whole(uv); assertContentEquals(floatArrayOf(1f, 1f, 1f, 0f, 0f, 1f, 0f, 0f), uv)
        val same = floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)                           // already the whole texture: unchanged
        TexQuad.whole(same); assertContentEquals(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f), same)
        val narrow = floatArrayOf(0.1f, 1f, 0.9f, 1f, 0.1f, 0f, 0.9f, 0f)                 // an image narrower than the texture
        TexQuad.whole(narrow); assertContentEquals(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f), narrow)
    }
}
