import os
import pickle
import numpy as np
import onnxruntime as ort

print("="*60)
print("  SYNTHETIC SIGNAL EVALUATION ENGINE (EXTREME TUNED) ONLINE  ")
print("="*60)

SELECTED_MODS = [
    'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
    'PAM4', 'WBFM', 'AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK'
]

dataset_path = "data/ultra_low.pkl"
if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset not found at {dataset_path}!")
    exit()

print("Loading benchmark demo signals...")
with open(dataset_path, 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

df = {}
for (mod, snr), data in raw_data.items():
    fixed_mod = mod
    if mod == '16QAM': fixed_mod = 'QAM16'
    elif mod == '64QAM': fixed_mod = 'QAM64'
    
    if fixed_mod in SELECTED_MODS:
        df[(fixed_mod, snr)] = data

snrs = sorted(list(set(k[1] for k in df.keys())))

print("Initializing Expert ONNX Runtime Sessions...")
models = {
    'extreme_low': ort.InferenceSession("models/best_model_v4_extreme_low_snr.onnx"),
    'ultra_low': ort.InferenceSession("models/best_model_v4_ultra_low_snr.onnx"),
    'low': ort.InferenceSession("models/best_model_v4_low_snr.onnx"),
    'mid_high': ort.InferenceSession("models/best_model_v4_mid_snr.onnx")
}

print("✓ All brains synced. Starting Pipeline Verification...\n")

# Replicating your training-stage window_size=3 micro-smoothing filter in NumPy
def numpy_suppress_noise_peaks(iq_sample, window_size=3):
    padding = window_size // 2
    smoothed = np.zeros_like(iq_sample)
    # Apply 3-point moving average to filter out massive -20dB noise spikes
    kernel = np.ones(window_size) / window_size
    smoothed[0, :] = np.convolve(iq_sample[0, :], kernel, mode='same')
    smoothed[1, :] = np.convolve(iq_sample[1, :], kernel, mode='same')
    return smoothed

global_correct = 0
global_total = 0
snr_report = {}
dataset_kernel = np.ones(5) / 5 # Inst Freq kernel from dataset.py

for target_snr in snrs:
    snr_correct = 0
    snr_total = 0
    
    if -20 <= target_snr <= -15:
        session = models['extreme_low']
        mode_str = "Extreme Low-SNR Expert"
    elif -15 < target_snr <= -10:
        session = models['ultra_low']
        mode_str = "Ultra Low-SNR Expert"
    elif -10 < target_snr <= -4:
        session = models['low']
        mode_str = "Low-SNR Expert"
    else:
        session = models['mid_high']
        mode_str = "Mid-High SNR Expert"
        
    for m in SELECTED_MODS:
        if (m, target_snr) in df:
            signals = df[(m, target_snr)]
            
            for i in range(len(signals)):
                raw_iq = signals[i]
                
                # STAGE 1: Execute Asymmetric Extreme Noise Suppression
                processed_iq = numpy_suppress_noise_peaks(raw_iq, window_size=3)
                
                I = processed_iq[0, :]
                Q = processed_iq[1, :]
                
                # STAGE 2: 5-Channel Feature Extraction Pipeline
                phase = np.arctan2(Q, I)
                phase_unwrapped = np.unwrap(phase)
                inst_freq = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])
                inst_freq = np.convolve(inst_freq, dataset_kernel, mode='same')
                if inst_freq.std() > 1e-8:
                    inst_freq = (inst_freq - inst_freq.mean()) / inst_freq.std()
                
                envelope = np.sqrt(I**2 + Q**2)
                if envelope.std() > 1e-8:
                    envelope = (envelope - envelope.mean()) / envelope.std()
                
                fft_mag = np.abs(np.fft.fft(I + 1j * Q))
                fft_mag = np.fft.fftshift(fft_mag)
                if fft_mag.std() > 1e-8:
                    fft_mag = (fft_mag - fft_mag.mean()) / fft_mag.std()
                
                # Structuring the 5-Channel feature space
                features = np.stack([I, Q, inst_freq, envelope, fft_mag], axis=0)
                features = np.expand_dims(features, axis=0).astype(np.float32)
                
                # Execute ONNX Runtime Evaluation
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: features})
                pred_class_idx = np.argmax(outputs[0], axis=1)[0]
                
                true_class_idx = SELECTED_MODS.index(m)
                if pred_class_idx == true_class_idx:
                    snr_correct += 1
                    global_correct += 1
                
                snr_total += 1
                global_total += 1
                
    acc = (snr_correct / snr_total) * 100
    snr_report[target_snr] = acc
    print(f"🚀 [SNR: {target_snr:3d} dB] -> Verified Accuracy: {acc:6.2f}% | Mode: {mode_str}")

print("\n" + "="*60)
print("  FINAL ACCURACY BENCHMARK REPORT  ")
print("="*60)
total_network_acc = (global_correct / global_total) * 100
print(f"🏆 System Combined Pipeline Accuracy: {total_network_acc:.2f}%")
print(f"🔥 Secured Extreme Noise Zone (-20 dB) Performance: {snr_report[-20]:.2f}%")
print("="*60)