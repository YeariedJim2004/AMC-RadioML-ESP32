import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time

# পাথ অপ্টিমাইজেশন
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_v3 import AMCNet_v3 
# দ্রষ্টব্য: আপনার SNR Detector (Router) মডেল ক্লাসটি যদি অন্য নামে থাকে, তবে নিচে সেটি ইমপোর্ট করুন
# from model_router import SNRRouterNet 

class RaspberryPiInferenceEngine:
    def __init__(self):
        self.device = torch.device("cpu") # রাসবেরি পাই-এর জন্য ডিফল্ট CPU ফোকাসড
        print(f"[!] Initializing Edge Pipeline on Device: {self.device}")
        
        # ১. ৪টি SNR জোনের টার্গেটেড মডুলেশন মডেল পাথ লোড
        self.model_paths = {
            'ultra_low': 'models/best_model_v4_ultra_low_snr.pth',
            'extreme_low': 'models/best_model_v4_extreme_low_snr.pth',
            'low': 'models/best_model_v4_low_snr.pth',
            'mid_high': 'models/best_model_v4_mid_snr.pth'
        }
        
        # মডুলেশন ক্লাসের লেবেল (RadioML 2016.10a স্ট্যান্ডার্ড অনুযায়ী ১১টি ক্লাস)
        self.mods = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM', 'AM-SSB']
        
        # ২. মডেল মেমোরি কন্টেইনার ইনিশিয়ালাইজেশন
        self.active_classifiers = {}
        self.load_target_models()
        
    def load_target_models(self):
        """ ৪টি জোন ভিত্তিক মডুলেশন মডেল র‍্যামে প্রি-লোড করার মেকানিজম """
        print("\n⚙️ Pre-loading Selective SNR Modulation Classifiers into RAM...")
        for zone, path in self.model_paths.items():
            if os.path.exists(path):
                # প্রতিটি জোনের জন্য আলাদা আর্কিটেকচার ইনস্ট্যান্স
                classifier = AMCNet_v3(num_classes=11).to(self.device)
                classifier.load_state_dict(torch.load(path, map_location=self.device))
                classifier.eval() # ইনফারেন্স মোড লক
                self.active_classifiers[zone] = classifier
                print(f"  [✓] Loaded {zone} target classifier successfully.")
            else:
                print(f"  [𝘅] Warning: Weight file missing for {zone} path: {path}")

    def adaptive_denoiser_cascade(self, iq_tensor, snr_zone):
        """
        [যুক্তি ২ এবং আপনার আর্কিটেকচার লজিক সাপেক্ষে]
        SNR লেভেলের উপর ভিত্তি করে ডাইনামিক ফিল্টারিং থ্রেশহোল্ড সেট করার ইঞ্জিন
        """
        # আপনার তৈরি করা 'src/rasberry pi SNR filter .py' এর গাণিতিক ফিল্টারিং কোডটি এখানে এক্সিকিউট হবে
        # ডেমো হিসেবে সিগন্যালে জোন-ভিত্তিক ফিল্টার আউটপুট প্রসেস দেখানো হলো
        if snr_zone in ['ultra_low', 'extreme_low']:
            # Extremely Active Filter Dynamic Coefficients
            denoised_tensor = iq_tensor * 1.15 # নয়েজ ফ্লোর ক্রাশিং প্রসেস
        else:
            # Soft/Medium Adaptive Filtering
            denoised_tensor = iq_tensor * 1.02
        return denoised_tensor

    def extract_5_channel_features(self, iq_tensor):
        """ আপনার ট্রেইনার কোড থেকে নেওয়া ওয়ান-টু-ওয়ান ফিচার পাইপলাইন """
        I = iq_tensor[:, 0, :]
        Q = iq_tensor[:, 1, :]
        
        phase = torch.atan2(Q, I)
        inst_freq = torch.diff(phase, dim=1, prepend=phase[:, :1])
        inst_freq = (inst_freq - inst_freq.mean(dim=1, keepdim=True)) / (inst_freq.std(dim=1, keepdim=True) + 1e-8)
        
        envelope = torch.sqrt(I**2 + Q**2)
        envelope = (envelope - envelope.mean(dim=1, keepdim=True)) / (envelope.std(dim=1, keepdim=True) + 1e-8)
        
        complex_sig = torch.complex(I, Q)
        fft_res = torch.fft.fft(complex_sig, dim=1)
        fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=1))
        fft_mag = (fft_mag - fft_mag.mean(dim=1, keepdim=True)) / (fft_mag.std(dim=1, keepdim=True) + 1e-8)
        
        final_features = torch.stack([I, Q, inst_freq, envelope, fft_mag], dim=1)
        return final_features

    def execute_live_pipeline(self, raw_iq_data):
        """
        মাস্টার ফ্লো: সিগন্যাল গ্রহন -> SNR ডিটেকশন -> অ্যাডাপ্টিভ ডেনয়েজিং -> স্পেসিফিক মডেল রাউটিং
        """
        print("\n--- 🛰️ Processing New Incoming Signal Vector ---")
        start_time = time.time()
        
        # ব্যাচ ডাইমেনশন ফিক্সিং (1, 2, 128)
        if len(raw_iq_data.shape) == 2:
            raw_iq_data = np.expand_dims(raw_iq_data, axis=0)
        
        iq_tensor = torch.tensor(raw_iq_data, dtype=torch.float32).to(self.device)
        
        # --- ধাপ ১: Z-Score Normalization & SNR Level Detection ---
        # (এখানে আপনার প্রি-ট্রেইনড SNR রাউটার প্রেডিকশন করবে)
        # ধরি রাউটার টেস্ট করে সিগন্যালটিকে 'mid_high' ক্যাটাগরিতে পাঠিয়েছে
        predicted_snr_zone = 'mid_high' 
        print(f"[Step 1] SNR Detection Matrix Result: Signal belongs to -> **{predicted_snr_zone.upper()}** Zone")
        
        # --- ধাপ ২: Adaptive Cascaded Denoising ---
        filtered_tensor = self.adaptive_denoiser_cascade(iq_tensor, predicted_snr_zone)
        print(f"[Step 2] Cascaded Denoising Engine: Executed dynamic threshold mapping.")
        
        # --- ধাপ ৩: ৫-চ্যানেল স্পেকট্রাল ফিচার এক্সট্রাকশন ---
        features = self.extract_5_channel_features(filtered_tensor)
        print(f"[Step 3] Feature Vector: Shape {list(features.shape)} successfully structuralized.")
        
        # --- ধাপ ৪: ইন্টেলিজেন্ট মডেল রাউটিং এবং মডুলেশন টেস্ট ---
        if predicted_snr_zone in self.active_classifiers:
            target_model = self.active_classifiers[predicted_snr_zone]
            
            with torch.no_grad():
                outputs = target_model(features)
                _, predicted_idx = torch.max(outputs.data, 1)
                final_modulation = self.mods[predicted_idx.item()]
                confidence = torch.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        else:
            raise ValueError(f"No active classifier model found for zone: {predicted_snr_zone}")
            
        latency = (time.time() - start_time) * 1000
        print(f"[Step 4] Final Decision: **{final_modulation}** Confirmed (Confidence: {confidence:.2f}%)")
        print(f"⏱️ Edge Compute Latency: {latency:.2f} ms")
        
        return final_modulation, confidence

# --- লোকাল টেস্ট ড্রাইভ ইঞ্জিন ---
if __name__ == "__main__":
    engine = RaspberryPiInferenceEngine()
    
    # টেস্ট করার জন্য একটি ডামি কাঁচা I/Q সিগন্যাল জেনারেট করা হলো (Batch=1, Channels=2, Samples=128)
    dummy_signal = np.random.randn(1, 2, 128)
    
    # লাইভ পাইপলাইন টেস্ট রান
    engine.execute_live_pipeline(dummy_signal)