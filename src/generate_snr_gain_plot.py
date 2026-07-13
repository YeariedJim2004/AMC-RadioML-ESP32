import matplotlib.pyplot as plt

# ১. ডেটা সেটআপ (আপনার স্ক্রিনশটের রেজাল্ট অনুযায়ী)
input_snr = [-18.0, -12.5, -7.0, 8.0]
snr_gain = [12.71, 9.92, 9.65, 16.79]
labels = ['-18', '-12.5', '-7', '+8']

# ২. ফিগার সাইজ এবং একাডেমিক স্টাইল সেটআপ
plt.figure(figsize=(12, 4.5), dpi=300) # হাই-রেজোলিউশন ৩০০ DPI (জার্নাল স্ট্যান্ডার্ড)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# ৩. গ্রিড এবং এক্সিস লাইন কনফিগারেশন
plt.grid(True, linestyle='-', alpha=0.3, color='#c0c0c0')
ax = plt.gca()
ax.set_facecolor('#ffffff') # ব্যাকগ্রাউন্ড সাদা
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.tick_params(direction='in', top=True, right=True, length=6)

# ৪. মেইন প্লট (Line with Square Markers)
plt.plot(input_snr, snr_gain, 
         color='#104E7A',          # ডিপ ব্লু থিম
         linestyle='-',            # সলিড লাইন
         linewidth=2.5,            # লাইনের পুরুত্ব
         marker='s',               # স্কয়ার মার্কার
         markersize=6,             # মার্কার সাইজ
         markerfacecolor='#ffffff',# মার্কারের ভেতরের রঙ সাদা
         markeredgewidth=2.5,      # মার্কারের বর্ডারের পুরুত্ব
         label='Avg. SNR Gain (dB)')

# ৫. প্রতিটি ডেটা পয়েন্টে টেক্সট লেবেল যুক্ত করা
for x, y in zip(input_snr, snr_gain):
    plt.text(x, y + 0.6, f"+{y:.2f} dB Gain", 
             ha='center', va='bottom', 
             fontsize=11, fontweight='bold', color='#000000')

# ৬. অক্ষ ও টাইটেল লেবেলিং
plt.title("Denoiser Quantitative SNR Gain Analysis across Input SNR Regimes", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Average Input SNR (dB)", fontsize=12, fontweight='bold', labelpad=8)
plt.ylabel("Average SNR Gain (dB)", fontsize=12, fontweight='bold', labelpad=8)

# ৭. এক্সিস লিমিট ও টিক্স নির্ধারণ
plt.xticks(input_snr, labels, fontsize=11)
plt.yticks(range(0, 22, 2), fontsize=11)
plt.xlim(-20, 10)
plt.ylim(0, 20)

# ৮. টেক্সট বক্স (Peak Gain Highlight)
props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#000000', alpha=1.0)
plt.text(1, 2, "Peak Gain: +16.79 dB @ +8 dB SNR", 
         fontsize=12, fontweight='bold', bbox=props, ha='center')

# ৯. লিজেন্ড এবং লেআউট অপ্টিমাইজেশন
plt.legend(loc='upper right', frameon=True, edgecolor='#000000', facecolor='#ffffff')
plt.tight_layout()

# ১০. ইমেজ ফাইল হিসেবে সেভ করা
output_path = "denoiser_snr_gain_analysis.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"[✓] গ্রাফটি সফলভাবে তৈরি হয়েছে এবং '{output_path}' নামে সেভ হয়েছে!")

# গ্রাফটি উইন্ডোতে পপ-আপ করে দেখানোর জন্য (ঐচ্ছিক)
plt.show()