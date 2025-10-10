package com.litemind.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.litemind.app.birefnet.BiRefNetEngineManager
import com.litemind.app.birefnet.computeStats
import com.litemind.app.birefnet.toGrayBitmap
import com.litemind.app.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var engineManager: BiRefNetEngineManager

    private var processedBitmap: Bitmap? = null
    private var lastOriginalSize: Pair<Int, Int>? = null
    private var modelInputSize: Pair<Int, Int> = 0 to 0

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                loadBitmapFromUri(uri)?.let { bitmap ->
                    val originalWidth = bitmap.width
                    val originalHeight = bitmap.height
                    val prepared = prepareBitmapForModel(bitmap)
                    if (prepared !== bitmap) {
                        bitmap.recycle()
                    }
                    processedBitmap = prepared
                    lastOriginalSize = originalWidth to originalHeight
                    binding.originalPreview.setImageBitmap(prepared)
                    binding.statusText.text =
                        getString(
                            R.string.status_image_loaded_scaled,
                            originalWidth,
                            originalHeight,
                            prepared.width,
                            prepared.height
                        )
                    binding.runInferenceButton.isEnabled =
                        engineManager.isInitialized && processedBitmap != null
                } ?: showToast(getString(R.string.error_decode_image))
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        engineManager = BiRefNetEngineManager(applicationContext)
        lifecycle.addObserver(engineManager)

        binding.pickImageButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
        binding.runInferenceButton.setOnClickListener {
            runInference()
        }
        binding.runInferenceButton.isEnabled = false
        binding.pickImageButton.isEnabled = false

        prepareEngine()
    }

    override fun onDestroy() {
        super.onDestroy()
        processedBitmap = null
        lastOriginalSize = null
    }

    private fun prepareEngine() {
        binding.statusText.text = getString(R.string.status_engine_loading)
        binding.progressBar.isVisible = true

        lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.IO) { engineManager.initialize() }
            }
            binding.progressBar.isVisible = false
            result.onSuccess {
                val (width, height) = engineManager.inputSize()
                modelInputSize = width to height
                binding.statusText.text =
                    getString(R.string.status_engine_ready, width, height)
                processedBitmap = processedBitmap?.let { current ->
                    val prepared = prepareBitmapForModel(current)
                    if (prepared !== current) {
                        binding.originalPreview.setImageBitmap(prepared)
                    }
                    prepared
                }
                lastOriginalSize?.let { (ow, oh) ->
                    processedBitmap?.let { prepared ->
                        binding.statusText.text = getString(
                            R.string.status_image_loaded_scaled,
                            ow,
                            oh,
                            prepared.width,
                            prepared.height
                        )
                    }
                }
                binding.pickImageButton.isEnabled = true
                binding.runInferenceButton.isEnabled = processedBitmap != null
            }.onFailure { throwable ->
                val message = throwable.message ?: getString(R.string.error_unknown)
                binding.statusText.text = getString(R.string.status_engine_error, message)
                showToast(message)
            }
        }
    }

    private fun runInference() {
        val bitmap = processedBitmap ?: run {
            showToast(getString(R.string.status_select_image))
            return
        }
        if (!engineManager.isInitialized) {
            showToast(getString(R.string.status_engine_not_ready))
            return
        }

        binding.progressBar.isVisible = true
        binding.runInferenceButton.isEnabled = false
        binding.pickImageButton.isEnabled = false
        binding.statusText.text = getString(R.string.status_inference_running)

        lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.Default) { engineManager.run(bitmap) }
            }

            binding.progressBar.isVisible = false
            binding.runInferenceButton.isEnabled = true
            binding.pickImageButton.isEnabled = true

            result.onSuccess { maskBytes ->
                val maskBitmap = maskBytes.toGrayBitmap(bitmap.width, bitmap.height)
                val stats = maskBytes.computeStats()
                binding.maskPreview.setImageBitmap(maskBitmap)
                binding.statusText.text = getString(
                    R.string.status_inference_done,
                    stats.min,
                    stats.max,
                    stats.mean
                )
            }.onFailure { throwable ->
                val message = throwable.message ?: getString(R.string.error_unknown)
                binding.statusText.text = getString(R.string.status_inference_error, message)
                showToast(message)
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)?.copy(Bitmap.Config.ARGB_8888, false)
            }
        } catch (ioe: IOException) {
            null
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun prepareBitmapForModel(source: Bitmap): Bitmap {
        val (targetWidth, targetHeight) = modelInputSize
        if (targetWidth <= 0 || targetHeight <= 0 ||
            (source.width == targetWidth && source.height == targetHeight)
        ) {
            return source
        }
        return Bitmap.createScaledBitmap(source, targetWidth, targetHeight, true)
    }
}
