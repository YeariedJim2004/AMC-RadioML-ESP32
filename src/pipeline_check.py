import os
import pickle
import numpy as np
import onnxruntime as ort

print("="*60)
print("   ADVANCED AMC PIPELINE DIAGNOSTIC SYSTEM (STEP 1-3)   ")
print("="*60)

# 1. Setup Configurations
dataset_path = "data/ultra_low.pkl"
model_path = "models/best_model_v4_ultra_low_snr.onnx"

if not os.path.exists(dataset_path) or not os.path.exists(model_path):
    print("❌ Error: Missing core dependency files (Dataset or ONNX model)!")
    exit()

# Load a specific sample for numerical processing comparison
with open(dataset_path, 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

detected_keys = list(raw_data.keys())
sample_key = detected_keys[0] # Target first track (e.g., ('BPSK', -15))
signals = raw_data[sample_key]
raw_signal = signals[0] # Shape: (2, 128)

print(f"📊 [DIAGNOSTIC TARGET]: Locked key {sample_key} | Total Samples: {len(signals)}")

# --- TEST 1: NUMERICAL PREPROCESSING AUDIT ---
print(f"\n🔍 [TEST 1] Auditing Numerical Preprocessing Pipelines...")

# 5-Channel Feature Extraction Copying your standard dataset.py logic
if_kernel = np.ones(5) / 5
I, Q = raw_signal[0, :], raw_signal[1, :]

# Basic Normalization Audit
mean_I, std_I = I.mean(), I.std()
print(f"✓ Raw I-Channel Mean: {mean_I:.4f} | Std: {std_I:.4f}")

phase = np.arctan2(Q, I)
phase_unwrapped = np.unwrap(phase)
inst_freq = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])
inst_freq = np.convolve(inst_freq, if_kernel, mode='same')
if inst_freq.std() > 1e-8:
    inst_freq_norm = (inst_freq - inst_freq.mean()) / inst_freq.std()

envelope = np.sqrt(I**2 + Q**2)
if envelope.std() > 1e-8:
    envelope_norm = (envelope - envelope.mean()) / envelope.std()

fft_mag = np.abs(np.fft.fft(I + 1j * Q))
fft_mag = np.fft.fftshift(fft_mag)
if fft_mag.std() > 1e-8:
    fft_mag_norm = (fft_mag - fft_mag.mean()) / fft_mag.std()

# Inspecting feature metrics for shape, limits and scaling mismatch tracking
print(f"✓ Extracted Ext_Freq Norm Vector Sample (First 3 entries): {inst_freq_norm[:3]}")
print(f"✓ Spectral Mag Norm Vector Sample (First 3 entries): {fft_mag_norm[:3]}")


# --- TEST 2: ONNX INFERENCE ENGINE & LOGIT INTEGRITY ---
print(f"\n🔍 [TEST 2] Verifying ONNX Inference & Logit Output... ")
features = np.stack([I, Q, inst_freq_norm, envelope_norm, fft_mag_norm], axis=0)
features_tensor = np.expand_dims(features, axis=0).astype(np.float32) # Mold to (1, 5, 128)

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
onnx_outputs = session.run(None, {input_name: features_tensor})
logits = onnx_outputs[0][0]

print(f"✓ ONNX Raw Probabilities/Logits Matrix Output:\n{logits}")
print(f"✓ Maximum Probability Index: {np.argmax(logits)} (Confidence: {np.max(logits)*100:.2f}%)")


# --- TEST 3: RAW CONFUSION MATRIX CALCULATION ---
print(f"\n🔍 [TEST 3] Generating Live Confusion Matrix Array ...")
MY_CLASSES = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4', 'WBFM', 'AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK']

# Initialize an 11x11 matrix
conf_matrix = np.zeros((11, 11), dtype=int)

for (mod, snr), data in raw_data.items():
    fixed_mod = mod
    if mod == '16QAM': fixed_mod = 'QAM16'
    elif mod == '64QAM': fixed_mod = 'QAM64'
    
    if fixed_mod in MY_CLASSES:
        true_idx = MY_CLASSES.index(fixed_mod)
        
        # Batch inference processing
        for sig in data:
            s_I, s_Q = sig[0, :], sig[1, :]
            
            # Local quick features
            s_phase = np.arctan2(s_Q, s_I)
            s_inst_freq = np.diff(np.unwrap(s_phase), prepend=np.unwrap(s_phase)[0])
            s_inst_freq = np.convolve(s_inst_freq, if_kernel, mode='same')
            if s_inst_freq.std() > 1e-8: s_inst_freq = (s_inst_freq - s_inst_freq.mean()) / s_inst_freq.std()
            
            s_envelope = np.sqrt(s_I**2 + s_Q**2)
            if s_envelope.std() > 1e-8: s_envelope = (s_envelope - s_envelope.mean()) / s_envelope.std()
            
            s_fft_mag = np.fft.fftshift(np.abs(np.fft.fft(s_I + 1j * s_Q)))
            if s_fft_mag.std() > 1e-8: s_fft_mag = (s_fft_mag - s_fft_mag.mean()) / s_fft_mag.std()
            
            feat = np.stack([s_I, s_Q, s_inst_freq, s_envelope, s_fft_mag], axis=0)
            feat_tensor = np.expand_dims(feat, axis=0).astype(np.float32)
            
            out = session.run(None, {input_name: feat_tensor})
            pred_idx = np.argmax(out[0], axis=1)[0]
            conf_matrix[true_idx, pred_idx] += 1

print("\n📊 Raw Confusion Matrix (Rows = True Classes, Columns = Predicted Classes):")
print(f"Class Sequence Order: {MY_CLASSES}")
print("-" * 75)
for row_idx, row in enumerate(conf_matrix):
    row_str = " ".join(f"{val:5d}" for val in row)
    print(f"{MY_CLASSES[row_idx]:7s} | {row_str}")
print("-" * 75)

print("\n" + "="*60)
print("   DIAGNOSTIC COMPLETED: ANALYZE PATTERNS ABOVE   ")
print("="*60)