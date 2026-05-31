import socket
import struct
import numpy as np
import onnxruntime as ort

# Network Configuration
RPI_PORT = 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", RPI_PORT))

print("Loading optimized ONNX models into execution memory...")
# Load exported ONNX runtime sessions
models = {
    'extreme_low': ort.InferenceSession("models/best_model_v4_extreme_low_snr.onnx"),
    'ultra_low': ort.InferenceSession("models/best_model_v4_ultra_low_snr.onnx"),
    'low': ort.InferenceSession("models/best_model_v4_low_snr.onnx"),
    'mid_high': ort.InferenceSession("models/best_model_v4_mid_snr.onnx")
}

classes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'BFSK', 'CPFSK', 'GFSK', 'AM-SSB', 'AM-DSB', 'WBFM']

print("Raspberry Pi Inference Core online. Waiting for ESP32 processed streams...\n")

while True:
    # Buffer payload size: 384 floats * 4 bytes = 1536 bytes
    data, addr = sock.recvfrom(1536)
    if len(data) < 1536:
        continue
        
    unpacked = struct.unpack(f'{128}f{128}f{128}f', data)
    I = np.array(unpacked[0:128])
    Q = np.array(unpacked[128:256])
    envelope = np.array(unpacked[256:384])
    
    # 1. Extract Instantaneous Frequency
    phase = np.arctan2(Q, I)
    inst_freq = np.diff(phase, prepend=phase[0])
    inst_freq = (inst_freq - np.mean(inst_freq)) / (np.std(inst_freq) + 1e-8)
    
    # Normalize Envelope
    envelope = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-8)
    
    # 2. Extract FFT Magnitude
    complex_sig = I + 1j * Q
    fft_res = np.abs(np.fft.fftshift(np.fft.fft(complex_sig)))
    fft_mag = (fft_res - np.mean(fft_res)) / (np.std(fft_res) + 1e-8)
    
    # Construct 5-Channel feature matrix [Shape: 1, 5, 128]
    final_features = np.stack([I, Q, inst_freq, envelope, fft_mag], axis=0)
    final_features = np.expand_dims(final_features, axis=0).astype(np.float32)
    
    # 3. Dynamic Model Routing based on estimated signal profile
    # Currently configured to route through the extreme low-SNR core for testing
    active_session = models['extreme_low'] 
    
    # Execute ONNX Inference
    inputs = {active_session.get_inputs()[0].name: final_features}
    outputs = active_session.run(None, inputs)
    
    # Calculate output probability matrix via Softmax
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx] * 100
    
    print(f"📡 [LIVE ONNX INFERENCE] Detected: {classes[predicted_idx]} | Confidence: {confidence:.2f}%")