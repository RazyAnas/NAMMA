package namma.capture

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/** Layouts in code (no XML): a few helpers. */
object Ui {
    const val BG = 0xFF101418.toInt(); const val CARD = 0xFF1C232B.toInt(); const val ACCENT = 0xFF2BB3A3.toInt()
    const val WARN = 0xFFE5484D.toInt(); const val TEXT = 0xFFECEFF1.toInt(); const val DIM = 0xFF9AA5AE.toInt()

    fun dp(c: Context, v: Int) = (v * c.resources.displayMetrics.density).toInt()

    fun column(c: Context, pad: Int = 16) = LinearLayout(c).apply {
        orientation = LinearLayout.VERTICAL; setPadding(dp(c, pad), dp(c, pad), dp(c, pad), dp(c, pad))
    }

    fun row(c: Context) = LinearLayout(c).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }

    fun text(c: Context, s: CharSequence, size: Float = 16f, color: Int = TEXT, bold: Boolean = false) = TextView(c).apply {
        text = s; textSize = size; setTextColor(color); if (bold) typeface = Typeface.DEFAULT_BOLD; setPadding(0, dp(c, 4), 0, dp(c, 4))
    }

    fun button(c: Context, s: String, color: Int = ACCENT, onClick: (View) -> Unit) = Button(c).apply {
        text = s; isAllCaps = false; textSize = 16f; setTextColor(Color.WHITE)
        background = GradientDrawable().apply { cornerRadius = dp(c, 10).toFloat(); setColor(color) }
        setPadding(dp(c, 16), dp(c, 12), dp(c, 16), dp(c, 12)); setOnClickListener(onClick)
    }

    fun card(c: Context) = column(c, 12).apply {
        background = GradientDrawable().apply { cornerRadius = dp(c, 12).toFloat(); setColor(CARD) }
    }

    /** Android 15+ draws every screen under the status bar, the navigation bar and the camera cut-out: keep these views'
     *  content clear of them (the insets are added to the padding each view already has). */
    fun insets(root: View, vararg views: View) {
        val base = views.map { intArrayOf(it.paddingLeft, it.paddingTop, it.paddingRight, it.paddingBottom) }
        root.setOnApplyWindowInsetsListener { _, w ->
            val l: Int; val t: Int; val r: Int; val b: Int
            if (Build.VERSION.SDK_INT >= 30) { val i = w.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout()); l = i.left; t = i.top; r = i.right; b = i.bottom }
            else { @Suppress("DEPRECATION") l = w.systemWindowInsetLeft; @Suppress("DEPRECATION") t = w.systemWindowInsetTop
                @Suppress("DEPRECATION") r = w.systemWindowInsetRight; @Suppress("DEPRECATION") b = w.systemWindowInsetBottom }
            views.forEachIndexed { k, v -> v.setPadding(base[k][0] + l, base[k][1] + t, base[k][2] + r, base[k][3] + b) }
            w
        }
        root.requestApplyInsets()
    }

    fun gap(c: Context, h: Int = 12) = View(c).apply { layoutParams = LinearLayout.LayoutParams(1, dp(c, h)) }

    fun wide(v: View, c: Context, top: Int = 8): View {
        v.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(c, top) }
        return v
    }
}
