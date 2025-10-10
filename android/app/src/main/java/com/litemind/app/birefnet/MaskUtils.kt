package com.litemind.app.birefnet

import android.graphics.Bitmap
import android.graphics.Color

fun ByteArray.toGrayBitmap(width: Int, height: Int): Bitmap {
    val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(width * height)
    for (index in indices) {
        val value = this[index].toInt() and 0xFF
        pixels[index] = Color.argb(255, value, value, value)
    }
    result.setPixels(pixels, 0, width, 0, 0, width, height)
    return result
}

data class MaskStats(
    val min: Int,
    val max: Int,
    val mean: Double
)

fun ByteArray.computeStats(): MaskStats {
    if (isEmpty()) return MaskStats(0, 0, 0.0)
    var min = 255
    var max = 0
    var sum = 0.0
    forEach { byte ->
        val value = byte.toInt() and 0xFF
        if (value < min) min = value
        if (value > max) max = value
        sum += value
    }
    val mean = sum / size.toDouble()
    return MaskStats(min = min, max = max, mean = mean)
}
