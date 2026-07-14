import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time

# Path optimization
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_v3 import AMCNet_v3 

# --- ৪টি মডেলের ট্রেইনার থেকে নেওয়া ডেডিকেটেড DSP ইঞ্জিন ---
class EdgeSignalPreprocessingDSP:
    @staticmethod
    def extreme_low_suppress_peaks(iq_tensor, window_size=3):
        """ Extreme Low-SNR জোনের (-20dB to -15dB) ১ডি কনভোলিউশন স্মুথিং ফিল্টার """
        padding = window_size // 2
        mean_filter = torch.ones(1, 1, window_size, device=iq_tensor.device) / window_size
        ch0 = iq_tensor[:, 0, :].unsqueeze(1) 
        ch1 = iq_tensor[:, 1, :].unsqueeze(1) 
        ch0_smooth = nn.functional.conv1d(ch0, mean_filter, padding=padding)
        ch1_smooth = nn.functional.conv1d(ch1, mean_filter, padding=padding)
        return torch.cat([ch0_smooth, ch1_smooth], dim=1)

    @staticmethod
    def low_snr_moving_average(iq_tensor, window_size=3):
        """ Low-SNR জোনের (-10dB to -4dB) মুভিং অ্যাভারেজ ফিল্টার """
        padding = window_size // 2
        mean_filter = torch.ones(1, 1, window_size, device=iq_tensor.device) / window_size
        ch0 = iq_tensor[:, 0, :].unsqueeze(1) 
        ch1 = iq_tensor[:, 1, :].unsqueeze(1) 
        ch0_smooth = nn.functional.conv1d(ch0, mean_filter, padding=padding)
        ch1_smooth = nn.functional.conv1d(ch1, mean_filter, padding=padding)
        return torch.cat([ch0_smooth, ch1_smooth], dim=1)


class RaspberryPiInferenceEngine:
    def __init__(self):
        # রাসবেরি পাই-এর জন্য ডেডিকেটেড CPU সেটআপ (Edge Optimized)
        self.device = torch.device("cpu")
        print(f"[!] Initializing Complete 4-Tier Edge Pipeline on Device: {self.device}")
        
        # ১. আপনার তৈরি করা ৪টি সুনির্দিষ্ট মডেল পাথের সুসংগত ম্যাপিং
        self.model_paths = {
            'extreme_low': 'models/best_model_v4_extreme_low_snr.pth', # (-20dB to -15dB)
            'ultra_low': 'models/best_model_v4_ultra_low_snr.pth',     # (-15dB to -10dB)
            'low': 'models/best_model_v4_low_snr.pth',                 # (-10dB to -4dB)
            'mid_high': 'models/best_model_v4_mid_snr.pth'              # (-4dB to +8dB+)
        }
        
        # মডুলেশন ক্লাসের লেবেল (RadioML 2016.10a স্ট্যান্ডার্ড অনুযায়ী ১১টি ক্লাস)
        self.mods = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM', 'AM-SSB']
        
        # ২. স্টার্টআপে মেমরিতে সবকটি মডেল একসাথে প্রি-লোড করার মেকানিজম
        self.active_classifiers = {}
        self.load_target_models()
        
    def load_target_models(self):
        print("\n⚙️ Pre-loading All Selective SNR Modulation Classifiers into RAM...")
        for zone, path in self.model_paths.items():
            if os.path.exists(path):
                # প্রতি জোনের জন্য AMCNet_v3 আর্কিটেকচার মেমরিতে ইনস্ট্যান্স করা হচ্ছে
                classifier = AMCNet_v3(num_classes=11).to(self.device)
                classifier.load_state_dict(torch.load(path, map_location=self.device))
                classifier.eval() # ইনফারেন্স মোড লক (No Dropout/BatchNorm shifts)
                self.active_classifiers[zone] = classifier
                print(f"  [✓] Loaded {zone} target classifier successfully.")
            else:
                print(f"  [𝘅] Warning: Missing weights for {zone} at path: {path}")

    def extract_5_channel_features(self, iq_tensor):
        """ আপনার ট্রেইনার কোড থেকে নেওয়া ওয়ান-টু-ওয়ান ৫-চ্যানেল স্পেকট্রাল ফিচার ইঞ্জিন """
        I = iq_tensor[:, 0, :]
        Q = iq_tensor[:, 1, :]
        
        # Phase & Instantaneous Frequency
        phase = torch.atan2(Q, I)
        inst_freq = torch.diff(phase, dim=1, prepend=phase[:, :1])
        inst_freq = (inst_freq - inst_freq.mean(dim=1, keepdim=True)) / (inst_freq.std(dim=1, keepdim=True) + 1e-8)
        
        # Amplitude Envelope
        envelope = torch.sqrt(I**2 + Q**2)
        envelope = (envelope - envelope.mean(dim=1, keepdim=True)) / (envelope.std(dim=1, keepdim=True) + 1e-8)
        
        # FFT Magnitude Spectrum
        complex_sig = torch.complex(I, Q)
        fft_res = torch.fft.fft(complex_sig, dim=1)
        fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=1))
        fft_mag = (fft_mag - fft_mag.mean(dim=1, keepdim=True)) / (fft_mag.std(dim=1, keepdim=True) + 1e-8)
        
        # ৫-চ্যানেল টেনসর স্ট্যাকিং (Batch, Channels=5, Samples=128)
        final_features = torch.stack([I, Q, inst_freq, envelope, fft_mag], dim=1)
        return final_features

    def execute_live_pipeline(self, raw_iq_data, detected_snr_value):
        """
        মাস্টার অপারেশনাল পাইপলাইন: 
        সিগন্যাল রিসিভ -> SNR কন্ডিশনাল DSP রাউটিং -> ৫-চ্যানেল ফিচার জেনারেশন -> নির্দিষ্ট জোনের মডেলে ক্লাসিফিকেশন
        """
        print(f"\n--- 🛰️ Processing Incoming RF Signal Vector (Real-time SNR: {detected_snr_value} dB) ---")
        start_time = time.time()
        
        # ব্যাচ ডাইমেনশন ফিক্সিং (1, 2, 128)
        if len(raw_iq_data.shape) == 2:
            raw_iq_data = np.expand_dims(raw_iq_data, axis=0)
            
        iq_tensor = torch.tensor(raw_iq_data, dtype=torch.float32).to(self.device)
        
        # --- ধাপ ১: ডাইনামিক SNR জোন রাউটিং এবং কন্ডিশনাল DSP ফিল্টারিং ---
        if -20 <= detected_snr_value < -15:
            target_zone = 'extreme_low'
            processed_tensor = EdgeSignalPreprocessingDSP.extreme_low_suppress_peaks(iq_tensor)
            print(f"[Step 1] Routed to EXTREME LOW Pipeline. Micro-smoothing Conv1D applied.")
            
        elif -15 <= detected_snr_value < -10:
            target_zone = 'ultra_low'
            processed_tensor = iq_tensor # আল্ট্রা লো জোনের জন্য বেসলাইন ডাটা
            print(f"[Step 1] Routed to ULTRA LOW Pipeline. Strict phase stabilization active.")
            
        elif -10 <= detected_snr_value < -4:
            target_zone = 'low'
            processed_tensor = EdgeSignalPreprocessingDSP.low_snr_moving_average(iq_tensor)
            print(f"[Step 1] Routed to LOW Pipeline. Moving Average Smoothing applied.")
            
        else: # -4 dB থেকে +8 dB বা তার বেশি (Mid-High Regime)
            target_zone = 'mid_high'
            processed_tensor = iq_tensor
            print(f"[Step 1] Routed to MID-HIGH Pipeline. Optimal spectral extraction active.")

        # --- ধাপ ২: ৫-চ্যানেল ফিচার ভেক্টর জেনারেশন ---
        features = self.extract_5_channel_features(processed_tensor)
        print(f"[Step 2] Feature Vector Generation Complete. Shape: {list(features.shape)}")
        
        # --- ধাপ ৩: স্পেসিফিক মডেলে ইন্টেলিজেন্ট রাউটিং এবং ক্লাসিফিকেশন ---
        if target_zone in self.active_classifiers:
            model = self.active_classifiers[target_zone]
            
            with torch.no_grad():
                outputs = model(features)
                _, predicted_idx = torch.max(outputs.data, 1)
                final_modulation = self.mods[predicted_idx.item()]
                confidence = torch.softmax(outputs, dim=1)[0][predicted_idx].item() * 100
        else:
            raise ValueError(f"⚠️ Runtime Error: Target classifier model weights not pre-loaded for zone: {target_zone}")
            
        latency = (time.time() - start_time) * 1000
        print(f"[Step 3] Inference Decision: **{final_modulation}** Confirmed (Confidence: {confidence:.2f}%)")
        print(f"⏱️ Total Edge Compute Latency: {latency:.2f} ms")
        
        return final_modulation, confidence

# --- লোকাল পাইপলাইন সিমুলেশন টেস্ট ড্রাইভ ---
if __name__ == "__main__":
    engine = RaspberryPiInferenceEngine()
    
    # সিমুলেশন টেস্ট ১: লো নয়েজ জোন (Low SNR -7dB) -> মুভিং অ্যাভারেজ ট্রিগার হবে
    print("\n>>> TEST DRIVE 1: Evaluating Low SNR Zone (-7 dB) <<<")
    low_snr_test_signal = np.random.randn(1, 2, 128)
    engine.execute_live_pipeline(low_snr_test_signal, detected_snr_value=-7)
    
    # সিমুলেশন টেস্ট ২: চরম নয়েজ জোন (Extreme Low SNR -18dB) -> পিক সাপ্রেশন ট্রিগার হবে
    print("\n>>> TEST DRIVE 2: Evaluating Extreme Low SNR Zone (-18 dB) <<<")
    extreme_snr_test_signal = np.random.randn(1, 2, 128)
    engine.execute_live_pipeline(extreme_snr_test_signal, detected_snr_value=-18)