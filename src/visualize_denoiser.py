import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# আপনার ফিল্টার মডিউলটি ইম্পোর্ট করা হচ্ছে
import importlib
try:
    rasberry_pi_filter = importlib.import_module("src.rasberry pi SNR filter ")
except ModuleNotFoundError:
    rasberry_pi_filter = importlib.import_module("rasberry pi SNR filter ")
ExtremelyActiveDenoiser = rasberry_pi_filter.ExtremelyActiveDenoiser

DATASET_PATH = "data/RML2016.10a_dict.pkl"

def main():
    print("📦 ডাটাসেট লোড হচ্ছে...")
    with open(DATASET_PATH, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')
    
    # টেস্টের জন্য একটি মডুলেশন এবং লো-SNR জোন বেছে নিচ্ছি (যেমন: QPSK বা BPSK)
    # আপনি চাইলে এখানে আপনার প্রজেক্টের যেকোনো মডুলেশন নাম দিতে পারেন
    target_mod = None
    available_mods = list(set([k[0] for k in raw_data.keys()]))
    
    # QPSK বা BPSK খোঁজা, না পেলে প্রথম মডুলেশনটি নেওয়া
    for m in ['QPSK', 'BPSK', available_mods[0]]:
        if m in available_mods:
            target_mod = m
            break
            
    target_snr = -10  # Standard Low SNR জোনের একটি রিডিং (-10 dB)
    detected_zone = 2 # Standard Low SNR এর জন্য জোন আইডি ২
    
    print(f"📊 {target_mod} সিগন্যাল ({target_snr} dB) প্রসেস করা হচ্ছে...")
    
    # নির্দিষ্ট সিগন্যাল স্যাম্পল এক্সট্রাক্ট করা
    signal_samples = raw_data[(target_mod, target_snr)]
    I_raw = signal_samples[0][0]
    Q_raw = signal_samples[0][1]
    
    # আমাদের ডেনয়েজার দিয়ে ফিল্টার করা
    I_clean, Q_clean = ExtremelyActiveDenoiser.process(I_raw, Q_raw, detected_zone)
    
    # ── গ্রাফ প্লট করা ──
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    time_axis = np.arange(128)
    
    # ১. Raw Time-Domain Signal
    axs[0, 0].plot(time_axis, I_raw, label='I (In-Phase)', color='crimson', alpha=0.7)
    axs[0, 0].plot(time_axis, Q_raw, label='Q (Quadrature)', color='navy', alpha=0.7)
    axs[0, 0].set_title(f"Raw Time-Domain Signal ({target_mod} @ {target_snr}dB)")
    axs[0, 0].set_xlabel("Time Samples")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # ২. Clean Time-Domain Signal
    axs[0, 1].plot(time_axis, I_clean, label='I (Clean)', color='green')
    axs[0, 1].plot(time_axis, Q_clean, label='Q (Clean)', color='darkorange')
    axs[0, 1].set_title("Denoised Time-Domain Signal (Phase Preserved)")
    axs[0, 1].set_xlabel("Time Samples")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # ৩. Raw Constellation Plot
    axs[1, 0].scatter(I_raw, Q_raw, color='purple', alpha=0.6, edgecolors='k')
    axs[1, 0].set_title("Raw Constellation (Noisy Scatter)")
    axs[1, 0].set_xlabel("In-Phase (I)")
    axs[1, 0].set_ylabel("Quadrature (Q)")
    axs[1, 0].grid(True)
    axs[1, 0].axhline(0, color='black',linewidth=0.5)
    axs[1, 0].axvline(0, color='black',linewidth=0.5)
    
    # ৪. Clean Constellation Plot
    axs[1, 1].scatter(I_clean, Q_clean, color='teal', alpha=0.8, edgecolors='k')
    axs[1, 1].set_title("Denoised Constellation (Convergence Matrix)")
    axs[1, 1].set_xlabel("In-Phase (I)")
    axs[1, 1].set_ylabel("Quadrature (Q)")
    axs[1, 1].grid(True)
    axs[1, 1].axhline(0, color='black',linewidth=0.5)
    axs[1, 1].axvline(0, color='black',linewidth=0.5)
    
    plt.tight_layout()
    
    # গ্রাফটি ইমেজ ফাইল হিসেবে সেভ করা
    output_image = "denoiser_visual_validation.png"
    plt.savefig(output_image, dpi=300)
    print(f"🎉 ১ম ধাপ সম্পন্ন! ভিজ্যুয়াল গ্রাফটি '{output_image}' নামে প্রজেক্ট ফোল্ডারে সেভ হয়েছে।")

if __name__ == "__main__":
    main()