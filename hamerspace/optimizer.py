
import os
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
import onnx

def optimize_model(model_path, quantize=False, prune=False, output_dir='optimized_model'):
    ext = os.path.splitext(model_path)[-1]
    
    if ext in ['.h5', '.tf']:
        if quantize:
            quantize_tf_model(model_path, output_dir)
        if prune:
            prune_tf_model(model_path, output_dir)
    elif ext == '.onnx':
        if quantize:
            quantize_onnx_model(model_path, output_dir)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

def quantize_tf_model(model_path, output_dir):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_quantized.tflite"), "wb") as f:
        f.write(tflite_model)
    print("Quantized model saved.")

def prune_tf_model(model_path, output_dir):
    model = tf.keras.models.load_model(model_path)
    pruned_model = prune_low_magnitude(model)
    
    os.makedirs(output_dir, exist_ok=True)
    pruned_model.save(os.path.join(output_dir, "model_pruned.h5"))
    print("Pruned model saved.")

def quantize_onnx_model(model_path, output_dir):
    model = onnx.load(model_path)
    onnx.save_model(model, os.path.join(output_dir, "model_quantized.onnx"))
    print("ONNX quantization complete.")
