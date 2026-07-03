import pickle
import numpy as np
import onnxruntime as ort
import os

# ── CONFIGURATION ─────────────────────────────────────────
DATASET_PATH = "data/RML2016.10a_dict.pkl"
ROUTER_MODEL_PATH = "models/best_snr_router.onnx"

SNR_ZONES = {
    0: "Ultra Low SNR (-20 to -15 dB)",
    1: "Extreme Low SNR (-15 to -10 dB)",
    2: "Standard Low SNR (-10 to -4 dB)",
    3: "Mid/High SNR (-4 dB+)"
}
# ──────────────────────────────────────────────────────────

def extract_snr_features(iq_sample):
    I = iq_sample[0]
    Q = iq_sample[1]
    
    # DSP Feature Engine
    envelope = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    inst_freq = np.diff(np.unwrap(phase), prepend=phase[0])
    fft_mag = np.abs(np.fft.fft(I + 1j * Q))
    
    # ৭টি স্ট্যাটিস্টিক্যাল মেট্রিক্স
    f1 = np.mean(I**2 + Q**2)                          # Signal Power
    f2 = np.var(I) + np.var(Q)                         # Raw Signal Variance
    f3 = np.mean(envelope)                             # Envelope Mean
    f4 = np.var(envelope)                              # Envelope Variance
    f5 = np.var(inst_freq)                             # Inst Freq Variance
    f6 = np.max(fft_mag)                               # FFT Peak
    f7 = np.max(fft_mag) / (np.mean(fft_mag) + 1e-8)   # FFT Peak-to-Average
    
    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)

def main():
    if not os.path.exists(ROUTER_MODEL_PATH):
        print(f"❌ Error: ONNX router model not found at {ROUTER_MODEL_PATH}")
        return

    print("🤖 ONNX Runtime ব্যবহার করে SNR রাউটার লোড করা হচ্ছে...")
    session = ort.InferenceSession(ROUTER_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    print(f"📦 ডাটাসেট লোড করা হচ্ছে: {DATASET_PATH}...")
    with open(DATASET_PATH, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')

    # জোন ভিত্তিক ট্র্যাকিং ডিকশনারি
    # format -> zone_id: [total_samples, correct_predictions]
    zone_stats = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
    
    print("\n🔍 সিগন্যাল এক্সট্রাকশন এবং লাইভ টেস্ট শুরু হচ্ছে...")
    
    for (mod, snr), data in raw_data.items():
        # SNR কন্ডিশন ম্যাপিং ভেরিফিকেশন
        if -20 <= snr < -15:
            true_zone = 0  # Ultra Low
        elif -15 <= snr < -10:
            true_zone = 1  # Extreme Low
        elif -10 <= snr < -4:
            true_zone = 2  # Standard Low
        elif -4 <= snr <= 20:
            true_zone = 3  # Mid/High
        else:
            continue
            
        for i in range(data.shape[0]):
            # ৭টি তথ্য উপাত্ত বের করা
            features = extract_snr_features(data[i])
            # ওএনএনএক্স এর শেপ মিলানো [1, 7]
            input_vector = np.expand_dims(features, axis=0)
            
            # মডেল প্রেডিকশন
            logits = session.run(None, {input_name: input_vector})[0]
            pred_zone = np.argmax(logits, axis=1)[0]
            
            # স্ট্যাটস আপডেট
            zone_stats[true_zone][0] += 1
            if pred_zone == true_zone:
                zone_stats[true_zone][1] += 1

    # ── 📊 ফাইনাল রিপোর্ট প্রিন্টিং ──
    print("\n" + "="*60)
    print("🏆  SNR DETECTION MODEL ACCURACY TEST REPORT  🏆")
    print("="*60)
    
    overall_total = 0
    overall_correct = 0
    
    for zone_id, counts in zone_stats.items():
        total, correct = counts[0], counts[1]
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        overall_total += total
        overall_correct += correct
        
        # ৯০-৯৫% টার্গেট চেক করার ভিজ্যুয়াল ইন্ডিকেটর
        status = "✅ PASSED" if accuracy >= 90.0 else "⚠️ BELOW TARGET"
        
        print(f"📂 {SNR_ZONES[zone_id]}")
        print(f"   ➔ Test Samples: {total} | Correct: {correct}")
        print(f"   ➔ Accuracy: {accuracy:.2f}%  [{status}]")
        print("-" * 60)
        
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0.0
    print(f"🎯 OVERALL ROUTER ACCURACY: {overall_accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()