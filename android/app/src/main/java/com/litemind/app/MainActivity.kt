package com.litemind.app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.litemind.app.databinding.ActivityMainBinding

/**
 * LiteMind Demo 页面：演示如何在 UI 层调用 JNI 推理接口。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.inferenceResult.text = getString(R.string.placeholder_inference_result)

        binding.runInference.setOnClickListener {
            binding.inferenceResult.text = NativeBridge.stringFromJNI()
        }
    }
}
