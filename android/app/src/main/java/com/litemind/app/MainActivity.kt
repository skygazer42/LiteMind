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

    private var originalBitmap: Bitmap? = null

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                loadBitmapFromUri(uri)?.let { bitmap ->
                    originalBitmap?.recycle()
                    originalBitmap = bitmap
                    binding.originalPreview.setImageBitmap(bitmap)
                    binding.statusText.text =
                        getString(R.string.status_image_loaded, bitmap.width, bitmap.height)
                    binding.runInferenceButton.isEnabled = engineManager.isInitialized
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
        originalBitmap?.recycle()
        originalBitmap = null
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
                binding.statusText.text =
                    getString(R.string.status_engine_ready, width, height)
                binding.pickImageButton.isEnabled = true
                binding.runInferenceButton.isEnabled = originalBitmap != null
            }.onFailure { throwable ->
                val message = throwable.message ?: getString(R.string.error_unknown)
                binding.statusText.text = getString(R.string.status_engine_error, message)
                showToast(message)
            }
        }
    }

    private fun runInference() {
        val bitmap = originalBitmap ?: run {
            showToast(getString(R.string.status_select_image))
            return
        }
        if (!engineManager.isInitialized) {
            showToast(getString(R.string.status_engine_not_ready))
            return
        }

        binding.progressBar.isVisible = true
        binding.runInferenceButton.isEnabled = false
        binding.statusText.text = getString(R.string.status_inference_running)

        lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.Default) { engineManager.run(bitmap) }
            }

            binding.progressBar.isVisible = false
            binding.runInferenceButton.isEnabled = true

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
}
