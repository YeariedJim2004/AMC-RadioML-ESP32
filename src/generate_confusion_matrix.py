import pickle
import numpy as np
import onnxruntime as ort
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATASET_PATH = "data/RML2016.10a_dict.pkl"
ROUTER_MODEL_PATH = "models/best_snr_router.onnx"

# 🎯 লেবেলের নাম ছোট করে প্রফেশনাল করা হলো যাতে কেটে না যায় (যেমনটা image_c24827.jpg এ হয়েছিল)
ZONE_NAMES = [
    "Ultra\nLow",
    "Extreme\nLow",
    "Standard\nLow",
    "Mid/High"
]

def z_score_normalize(arr):
    std = arr.std()
    return (arr - arr.mean()) / (std + 1e-8)

def generate_5channel_tensor(iq_sample):
    I, Q = iq_sample[0], iq_sample[1]
    envelope = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    inst_freq = np.diff(np.unwrap(phase), prepend=phase[0])
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft(I + 1j * Q)))
    
    return np.vstack([
        z_score_normalize(I)[np.newaxis, :],
        z_score_normalize(Q)[np.newaxis, :],
        z_score_normalize(inst_freq)[np.newaxis, :],
        z_score_normalize(envelope)[np.newaxis, :],
        z_score_normalize(fft_mag)[np.newaxis, :]
    ]).astype(np.float32)

def main():
    if not os.path.exists(ROUTER_MODEL_PATH):
        print("❌ Error: ONNX Model missing! models/best_snr_router.onnx খুঁজে পাওয়া যায়নি।")
        return
        
    session = ort.InferenceSession(ROUTER_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    print("📦 ডাটাসেট লোড হচ্ছে...")
    with open(DATASET_PATH, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')

    y_true = []
    y_pred = []
    
    print("🔍 ওএনএনএক্স মডেল দিয়ে সিগন্যাল প্রেডিকশন এবং জোন ম্যাপিং চলছে...")
    for (mod, snr), data in raw_data.items():
        if -20 <= snr < -15: true_zone = 0
        elif -15 <= snr < -10: true_zone = 1
        elif -10 <= snr < -4: true_zone = 2
        elif -4 <= snr <= 20: true_zone = 3
        else: continue
            
        for i in range(data.shape[0]):
            features_5ch = generate_5channel_tensor(data[i])
            input_vector = np.expand_dims(features_5ch, axis=0)
            
            logits = session.run(None, {input_name: input_vector})[0]
            pred_zone = np.argmax(logits, axis=1)[0]
            
            y_true.append(true_zone)
            y_pred.append(pred_zone)

    # ── 📊 CONFUSION MATRIX GRAPH GENERATION ──
    print("🎨 পাবলিকেশন-কোয়ালিটি সুন্দর হিটম্যাপ তৈরি হচ্ছে...")
    cm = confusion_matrix(y_true, y_pred)
    
    # শতকরা হারে কনভার্ট করা (Normalized by True Labels Rows)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # গ্রাফের সাইড স্পেসিং সুন্দর করার জন্য সাইজ সামান্য বাড়িয়ে ৯.৫ করা হলো
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    
    # সুন্দর প্রফেশনাল ব্লু শেড কালারম্যাপ ব্যবহার
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=ZONE_NAMES, 
        yticklabels=ZONE_NAMES,
        cbar_kws={'label': 'Classification Accuracy (%)'},
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title("Confusion Matrix: 1D-CNN Based SNR Zone Router", fontsize=14, pad=20, weight='bold')
    plt.xlabel("Predicted SNR Zone", fontsize=12, labelpad=12, weight='bold')
    plt.ylabel("True SNR Zone", fontsize=12, labelpad=12, weight='bold')
    
    # টেক্সট ওভারল্যাপ বন্ধ করতে রোটেশন সোজা (0) করা হলো
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # 🎯 চারপাশের মার্জিন এবং প্যাডিং ম্যানুয়ালি অ্যাডজাস্ট করা হলো যেন কোনো লেখা না কাটে
    plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
    plt.tight_layout()
    
    # ডিরেক্টরি চেক করে ইমেজ সেভ করা
    os.makedirs("models", exist_ok=True)
    output_image = "models/snr_router_confusion_matrix.png"
    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"✅ ঝকঝকে গ্রাফটি সফলভাবে তৈরি হয়েছে এবং সেভ হয়েছে: {output_image}")

if __name__ == "__main__":
    main()