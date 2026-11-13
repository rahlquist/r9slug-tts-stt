"""Bake the Parakeet TDT 0.6B v3 INT8 ONNX model into the image (real files, no symlinks)."""
import onnx_asr

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "/models/parakeet", quantization="int8")
print("parakeet prewarm OK:", type(model).__module__)
