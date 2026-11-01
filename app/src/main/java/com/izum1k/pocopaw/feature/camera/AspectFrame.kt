package com.izum1k.pocopaw.feature.camera

import android.graphics.RuntimeShader
import android.os.Build
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FiniteAnimationSpec
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Stable
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asComposeRenderEffect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.dp
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.ui.theme.YoinMotion
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Every marble sensor frames 4:3; as portrait height / width. */
const val SENSOR_RATIO = 4f / 3f

val AspectRatio.portraitRatio: Float get() = width.toFloat() / height

/**
 * The framing seen through the viewfinder, in two parts that never move together:
 * [crop] is the ratio the seam dissolve frames inside the sensor's own 4:3 picture, [scale]
 * blows that picture up so the framed band fills the width. A switch always passes through the
 * 4:3 picture at its own size: grow back to it, let the seam close or open to the new framing,
 * then blow the new framing up.
 */
@Stable
class AspectFrame(ratio: Float) {
    val crop = Animatable(ratio)
    val scale = Animatable(scaleFor(ratio))

    /** The framed band inside the 4:3 picture, as fractions of its width and height. */
    val bandWidth: Float get() = min(1f, SENSOR_RATIO / crop.value)
    val bandHeight: Float get() = min(1f, crop.value / SENSOR_RATIO)

    /** The viewfinder's height over its width right now. */
    val heightRatio: Float get() = SENSOR_RATIO * bandHeight * scale.value

    /** How far the picture is blown up toward the tallest framing (0 at the 4:3 picture's size). */
    fun blowUp(widest: Float): Float {
        val full = scaleFor(widest) - 1f
        return if (full <= 0f) 0f else ((scale.value - 1f) / full).coerceIn(0f, 1f)
    }

    suspend fun switchTo(ratio: Float, spatial: FiniteAnimationSpec<Float>, seam: FiniteAnimationSpec<Float>) = coroutineScope {
        if (abs(crop.value - ratio) > 1e-4f) {
            val shrink = launch { scale.animateTo(1f, spatial) }
            scale.awaitNear(1f)
            val close = launch { crop.animateTo(ratio, seam) }
            crop.awaitNear(ratio)
            shrink.join()
            launch { scale.animateTo(scaleFor(ratio), spatial) }
            close.join()
        } else {
            scale.animateTo(scaleFor(ratio), spatial)
        }
    }

    /** Lets the next phase start while this one settles its last [HANDOFF] of the way. */
    private suspend fun Animatable<Float, *>.awaitNear(target: Float) {
        val from = value
        snapshotFlow { value }.first { abs(target - it) <= abs(target - from) * HANDOFF }
    }

    companion object {
        private const val HANDOFF = .08f

        fun scaleFor(ratio: Float): Float = max(1f, ratio / SENSOR_RATIO)
    }
}

@Composable
fun rememberAspectFrame(aspect: AspectRatio): AspectFrame {
    val frame = remember { AspectFrame(aspect.portraitRatio) }
    val spatial = YoinMotion.slowSpatialSpec<Float>()
    val seam = remember { YoinMotion.artworkDissolveSpring() }
    LaunchedEffect(aspect) { frame.switchTo(aspect.portraitRatio, spatial, seam) }
    return frame
}

/**
 * Yoin's seam dissolve (PiliPaw seam/SeamHalftone, curve C) turned to face the band's edges:
 * past the framed band the picture breaks into a staggered halftone whose dots shrink to nothing
 * over [AspectSeamTokens.Ramp], with Yoin's ragged front. Put it on the 4:3 picture before it is
 * scaled; at rest with nothing masked it draws nothing.
 */
fun Modifier.aspectSeam(frame: AspectFrame): Modifier =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) aspectSeamShader(frame) else aspectSeamFade(frame)

private fun Modifier.aspectSeamShader(frame: AspectFrame): Modifier {
    var shader: RuntimeShader? = null
    return graphicsLayer {
        val bw = frame.bandWidth
        val bh = frame.bandHeight
        if (bw >= 0.9999f && bh >= 0.9999f) {
            renderEffect = null
            return@graphicsLayer
        }
        val s = shader ?: RuntimeShader(AspectSeamTokens.AGSL).also { shader = it }
        val left = size.width * (1f - bw) / 2f
        val top = size.height * (1f - bh) / 2f
        val pitch = AspectSeamTokens.Pitch.toPx()
        val columns = max(4, (size.width / pitch).toInt())
        val cellWidth = size.width / columns
        s.setFloatUniform("size", size.width, size.height)
        s.setFloatUniform("cell", cellWidth, cellWidth * .866f)
        s.setFloatUniform("band", left, top, size.width - left, size.height - top)
        s.setFloatUniform("ramp", AspectSeamTokens.Ramp.toPx())
        s.setFloatUniform("front", AspectSeamTokens.Front.toPx())
        renderEffect = android.graphics.RenderEffect.createRuntimeShaderEffect(s, "content").asComposeRenderEffect()
    }
}

/** Below API 33 there is no runtime shader: the same band, edged by a plain fade to black. */
private fun Modifier.aspectSeamFade(frame: AspectFrame): Modifier = drawWithContent {
    drawContent()
    val bw = frame.bandWidth
    val bh = frame.bandHeight
    val ramp = AspectSeamTokens.Ramp.toPx()
    if (bw < 0.9999f) {
        val edge = size.width * (1f - bw) / 2f
        listOf(edge to edge - ramp, size.width - edge to size.width - edge + ramp).forEach { (inner, outer) ->
            val x0 = min(inner, outer)
            drawRect(
                Brush.horizontalGradient(listOf(Color.Transparent, Color.Black), startX = inner, endX = outer),
                topLeft = Offset(x0, 0f),
                size = Size(abs(outer - inner), size.height),
            )
        }
        drawRect(Color.Black, size = Size(max(0f, edge - ramp), size.height))
        drawRect(Color.Black, topLeft = Offset(size.width - edge + ramp, 0f), size = Size(max(0f, edge - ramp), size.height))
    }
    if (bh < 0.9999f) {
        val edge = size.height * (1f - bh) / 2f
        listOf(edge to edge - ramp, size.height - edge to size.height - edge + ramp).forEach { (inner, outer) ->
            val y0 = min(inner, outer)
            drawRect(
                Brush.verticalGradient(listOf(Color.Transparent, Color.Black), startY = inner, endY = outer),
                topLeft = Offset(0f, y0),
                size = Size(size.width, abs(outer - inner)),
            )
        }
        drawRect(Color.Black, size = Size(size.width, max(0f, edge - ramp)))
        drawRect(Color.Black, topLeft = Offset(0f, size.height - edge + ramp), size = Size(size.width, max(0f, edge - ramp)))
    }
}

private object AspectSeamTokens {
    /** Coarser than PiliPaw's 6dp list print: the viewfinder is one big picture. */
    val Pitch = 9.dp

    /** The dots go from whole print to nothing over this, outward from the band's edge. */
    val Ramp = 56.dp

    /** Yoin's ragged front (bend plus per-dot jitter). */
    val Front = 10.dp

    /** Dots start whole (the lattice's covering radius) at the band's edge, so there is no seam line. */
    const val DotGamma = .8f

    val AGSL = """
        uniform shader content;
        uniform float2 size;
        uniform float2 cell;
        uniform float4 band;
        uniform float ramp;
        uniform float front;

        const float TAU = 6.2831853;

        float hash(float2 p) {
            return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.547);
        }

        // How far outside the band, px (negative inside).
        float outside(float2 p) {
            return max(max(band.x - p.x, p.x - band.z), max(band.y - p.y, p.y - band.w));
        }

        half4 main(float2 xy) {
            float d = outside(xy);
            if (d <= 0.0) return content.eval(xy);
            if (d > ramp + front + cell.y * 2.0) return half4(0.0);
            float coverage = 0.0;
            float rowBase = floor(xy.y / cell.y - 0.5);
            for (int i = -1; i <= 2; i++) {
                float row = rowBase + float(i);
                float stagger = mod(row, 2.0) >= 1.0 ? 0.5 : 0.0;
                float ly = (row + 0.5) * cell.y;
                float columnBase = floor(xy.x / cell.x - 0.5 - stagger);
                for (int j = -1; j <= 2; j++) {
                    float column = columnBase + float(j);
                    float lx = (column + 0.5 + stagger) * cell.x;
                    float2 site = float2(lx, ly);
                    float u = lx / size.x;
                    float v = ly / size.y;
                    float bend = 0.10 * sin((u * 0.9 + v * 1.7) * TAU) + 0.05 * sin((v * 2.3 - u * 0.4 + 0.3) * TAU)
                        + (hash(float2(column, row)) - 0.5) * 0.14;
                    float shifted = outside(site) - (bend / 0.22) * front;
                    if (shifted <= 0.0) {
                        coverage = max(coverage, clamp(length(cell) * 0.53 - distance(xy, site) + 0.5, 0.0, 1.0));
                        continue;
                    }
                    float t = 1.0 - clamp(shifted / ramp, 0.0, 1.0);
                    if (t <= 0.0) continue;
                    float r = length(cell) * 0.53 * pow(t, $DotGamma);
                    float2 centre = site + float2(
                        cell.x * 0.42 * sin((v * 1.4 + u * 0.45 - t * 0.8) * TAU),
                        cell.y * 0.38 * sin((v * 0.7 - u * 1.3 + t * 0.65) * TAU)
                    ) * sin(t * 3.1415927) * sqrt(1.0 - t) * 0.6;
                    coverage = max(coverage, clamp(r - distance(xy, centre) + 0.5, 0.0, 1.0));
                }
            }
            if (coverage <= 0.0) return half4(0.0);
            return content.eval(xy) * coverage;
        }
    """.trimIndent()
}
