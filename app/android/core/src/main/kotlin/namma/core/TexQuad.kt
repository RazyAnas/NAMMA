package namma.core

/**
 * Texture coordinates for drawing a WHOLE camera texture (16 Nov). ARCore gives the corners of its CPU image in texture
 * coordinates; that image is 4:3 on most phones while the texture (the video picture) is 16:9, so those corners lie outside
 * 0..1 (v from -1/6 to 7/6). Drawn like that, the video was the picture squeezed to 75 % of its height with the edge rows
 * smeared over the top and bottom eighth (owner's first uploaded scan, 16 Nov). The corners still say how the picture is
 * turned and mirrored, so each one is moved to the texture corner it points towards: the whole texture, same orientation.
 */
object TexQuad {
    /** [uv] = 4 corners as u, v pairs; rewritten in place to the texture's own corners (0 or 1 on each axis). */
    fun whole(uv: FloatArray) {
        for (axis in 0..1) {
            var lo = Float.MAX_VALUE; var hi = -Float.MAX_VALUE
            for (i in 0 until 4) { val x = uv[2 * i + axis]; if (x < lo) lo = x; if (x > hi) hi = x }
            val mid = (lo + hi) / 2f
            for (i in 0 until 4) uv[2 * i + axis] = if (uv[2 * i + axis] > mid) 1f else 0f
        }
    }
}
