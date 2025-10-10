package com.litemind.app.birefnet

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.litemind.app.NativeBridge
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.locks.ReentrantLock

class BiRefNetEngineManager(
    private val context: Context,
    private val threads: Int = DEFAULT_THREADS,
    private val assetPath: String = DEFAULT_ASSET_PATH
) : DefaultLifecycleObserver {

    @Volatile
    private var handle: Long = 0L

    private val stateLock = ReentrantLock()
    private var activeInferences: Int = 0
    private var pendingDestroy: Boolean = false

    val isInitialized: Boolean
        get() = handle != 0L

    @Throws(IOException::class)
    fun initialize() {
        stateLock.lock()
        try {
            ensureHandleLocked()
        } finally {
            stateLock.unlock()
        }
    }

    fun release() {
        var handleToDestroy = 0L
        stateLock.lock()
        try {
            if (handle == 0L) {
                pendingDestroy = false
                return
            }
            if (activeInferences > 0) {
                pendingDestroy = true
                return
            }
            handleToDestroy = handle
            handle = 0L
            pendingDestroy = false
        } finally {
            stateLock.unlock()
        }
        if (handleToDestroy != 0L) {
            NativeBridge.destroyEngine(handleToDestroy)
        }
    }

    override fun onDestroy(owner: LifecycleOwner) {
        release()
    }

    fun run(bitmap: Bitmap): ByteArray {
        val activeHandle: Long
        stateLock.lock()
        try {
            if (pendingDestroy) {
                throw IllegalStateException("Engine is shutting down")
            }
            activeHandle = ensureHandleLocked()
            activeInferences++
        } finally {
            stateLock.unlock()
        }

        val prepared =
            if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
                ?: throw IllegalStateException("Failed to convert bitmap to ARGB_8888")

        var handleToDestroy = 0L
        return try {
            NativeBridge.runInference(activeHandle, prepared)
        } finally {
            stateLock.lock()
            try {
                activeInferences--
                if (activeInferences == 0 && pendingDestroy) {
                    handleToDestroy = handle
                    handle = 0L
                    pendingDestroy = false
                }
            } finally {
                stateLock.unlock()
            }
            if (handleToDestroy != 0L) {
                NativeBridge.destroyEngine(handleToDestroy)
            }
        }
    }

    fun inputSize(): Pair<Int, Int> {
        val activeHandle: Long
        stateLock.lock()
        try {
            if (pendingDestroy) {
                throw IllegalStateException("Engine is shutting down")
            }
            activeHandle = ensureHandleLocked()
        } finally {
            stateLock.unlock()
        }
        val dims = NativeBridge.getModelInputSize(activeHandle)
        if (dims.size < 2) return 0 to 0
        return dims[0] to dims[1]
    }

    private fun ensureHandleLocked(): Long {
        if (handle == 0L) {
            val modelFile = ensureModelFile()
            handle = NativeBridge.createEngine(modelFile.absolutePath, threads)
            pendingDestroy = false
        }
        return handle
    }

    @Throws(IOException::class)
    private fun ensureModelFile(): File {
        val target = File(context.filesDir, assetPath)
        if (target.exists()) return target
        target.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        return target
    }

    companion object {
        private const val DEFAULT_ASSET_PATH = "birefnet/birefnet_w8_nostat.mnn"
        private const val DEFAULT_THREADS = 4
    }
}
