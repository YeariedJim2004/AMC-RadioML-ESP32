import numpy as np
import scipy.signal as signal

class ExtremelyActiveDenoiser:
    """
    SNR লেভেলের ওপর ভিত্তি করে সিগন্যালের নয়েজ ধ্বংস করার আল্ট্রা-অ্যাক্টিভ মডিউল।
    এটি I এবং Q চ্যানেলের ফ্রিকোয়েন্সি এবং এমপ্লিচিউডকে ডাইনামিকালি ফিল্টার করে।
    """
    
    @staticmethod
    def process(I, Q, detected_zone):
        """
        I: np.array (length 128)
        Q: np.array (length 128)
        detected_zone: int (0: Ultra Low, 1: Extreme Low, 2: Standard Low, 3: Mid/High)
        """
        # ১. রাউটার মডেলের সিদ্ধান্তের ওপর ভিত্তি করে ফিল্টারের তীব্রতা (Agressive Tuning) নির্ধারণ
        if detected_zone == 0:    # Ultra Low SNR (-20 to -15 dB) -> চরম ফিল্টারিং
            filter_order = 5
            cutoff_freq = 0.25     # শুধুমাত্র কোর লো-ফ্রিকোয়েন্সি সিগন্যাল পাস করবে, বাকি সব ব্লক
            wiener_noise_floor = 0.45
            
        elif detected_zone == 1:  # Extreme Low SNR (-15 to -10 dB) -> উচ্চ ফিল্টারিং
            filter_order = 4
            cutoff_freq = 0.35
            wiener_noise_floor = 0.30
            
        elif detected_zone == 2:  # Standard Low SNR (-10 to -4 dB) -> মাঝারি ফিল্টারিং
            filter_order = 3
            cutoff_freq = 0.50
            wiener_noise_floor = 0.15
            
        else:                     # Mid/High SNR (-4 dB+) -> লাইট ফিল্টারিং (যাতে সিগন্যাল ডিটেইল না হারায়)
            filter_order = 2
            cutoff_freq = 0.75
            wiener_noise_floor = 0.02

        # ২. ফেজ-প্রিজার্ভিং বাটারওয়ার্থ লো-পাস ফিল্টার (Butterworth Filter)
        # এটি সিগন্যালের ফেজ শিফট না ঘটিয়ে হাই-ফ্রিকোয়েন্সি নয়েজ ব্লেড কেটে ফেলে
        b, a = signal.butter(filter_order, cutoff_freq, btype='low')
        
        # filtfilt ব্যবহার করায় সিগন্যাল সামনে এবং পেছনে দুইবার ফিল্টার হয়, ফলে Phase Distortion শূন্য হয়
        I_lowpass = signal.filtfilt(b, a, I)
        Q_lowpass = signal.filtfilt(b, a, Q)

        # ৩. অ্যাডাপ্টিভ উইনার ফিল্টারিং (Adaptive Wiener Filter)
        # এটি লোকাল ভ্যারিয়েন্স হিসাব করে যেখানে নয়েজ বেশি সেখানে সাপ্রেস করে, যেখানে সিগন্যাল সবল সেখানে ছেড়ে দেয়
        I_clean = ExtremelyActiveDenoiser._wiener_filter(I_lowpass, wiener_noise_floor)
        Q_clean = ExtremelyActiveDenoiser._wiener_filter(Q_lowpass, wiener_noise_floor)
        
        return I_clean.astype(np.float32), Q_clean.astype(np.float32)

    @staticmethod
    def _wiener_filter(x, noise_floor):
        # উইনার ফিল্টারের ইন্টারনাল ম্যাথ: লোকাল গড় এবং ভ্যারিয়েন্স ট্র্যাক করা
        window_size = 7
        local_mean = np.convolve(x, np.ones(window_size)/window_size, mode='same')
        local_var = np.convolve(x**2, np.ones(window_size)/window_size, mode='same') - local_mean**2
        local_var = np.maximum(local_var, 0)
        
        # নয়েজ এস্টিমেশন
        noise_var = np.mean(local_var) * noise_floor
        
        # ফিল্টার গেইন ম্যাট্রিক্স
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = local_var / (local_var + noise_var)
            gain = np.where(np.isnan(gain), 0, gain)
            
        return local_mean + gain * (x - local_mean)