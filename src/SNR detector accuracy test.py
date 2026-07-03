import pickle
import numpy as np
import onnxruntime as ort
import os

DATASET_PATH = "data/RML2016.10a_dict.pkl"
ROUTER_MODEL_PATH = "models/best_snr_router.onnx"

SNR_ZONES = {
    0: "Ultra Low SNR (-20 to -15 dB)",
    1: "Extreme Low SNR (-15 to -10 dB)",
    2: "Standard Low SNR (-10 to -4 dB)",
    3: "Mid/High SNR (-4 dB+)"
}

def z_score_normalize(arr):
    std = arr.std()
    return (arr - arr.mean()) / (std + 1e-8)

def generate_5channel_tensor(iq_sample):
    I = iq_sample[0]
    Q = iq_sample[1]
    
    envelope = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    inst_freq = np.diff(np.unwrap(phase), prepend=phase[0])
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft(I + 1j * Q)))
    
    iq_5ch = np.vstack([
        z_score_normalize(I)[np.newaxis, :],
        z_score_normalize(Q)[np.newaxis, :],
        z_score_normalize(inst_freq)[np.newaxis, :],
        z_score_normalize(envelope)[np.newaxis, :],
        z_score_normalize(fft_mag)[np.newaxis, :]
    ]).astype(np.float32)
    
    return iq_5ch

def main():
    if not os.path.exists(ROUTER_MODEL_PATH):
        print("❌ Error: ONNX Model missing!")
        return
        
    session = ort.InferenceSession(ROUTER_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    with open(DATASET_PATH, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')

    zone_stats = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
    
    print("🔍 1D-CNN আনবায়াসড জোন-ভিত্তিক টেস্ট শুরু হচ্ছে...")
    for (mod, snr), data in raw_data.items():
        if -20 <= snr < -15: true_zone = 0
        elif -15 <= snr < -10: true_zone = 1
        elif -10 <= snr < -4: true_zone = 2
        elif -4 <= snr <= 20: true_zone = 3
        else: continue
            
        for i in range(data.shape[0]):
            features_5ch = generate_5channel_tensor(data[i])
            input_vector = np.expand_dims(features_5ch, axis=0) # [1, 5, 128]
            
            logits = session.run(None, {input_name: input_vector})[0]
            pred_zone = np.argmax(logits, axis=1)[0]
            
            zone_stats[true_zone][0] += 1
            if pred_zone == true_zone:
                zone_stats[true_zone][1] += 1

    print("\n" + "="*60)
    print("🏆       🌟 1D-CNN SNR ROUTER FINAL SYSTEM REPORT 🌟       🏆")
    print("="*60)
    for zone_id, counts in zone_stats.items():
        total, correct = counts[0], counts[1]
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        print(f"📂 {SNR_ZONES[zone_id]} ➔ Accuracy: {accuracy:.2f}%")
    print("==============================================================")

if __name__ == "__main__":
    main()