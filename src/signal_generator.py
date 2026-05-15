import numpy as np

class AMCSignalGenerator:
    def __init__(self, num_samples=1024):
        self.num_samples = num_samples
        # 11 types of modulations as per project outline
        self.modulations = [
            'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 
            'PAM4', 'WBFM', 'AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK'
        ]

    def add_noise(self, signal, snr_db):
        snr_linear = 10**(snr_db / 10)
        sig_pwr = np.mean(np.abs(signal)**2)
        noise_pwr = sig_pwr / snr_linear
        noise = np.sqrt(noise_pwr/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
        return signal + noise

    # --- Digital Modulations ---
    def generate_bpsk(self):
        return 2 * np.random.randint(0, 2, self.num_samples) - 1

    def generate_qpsk(self):
        data = np.random.randint(0, 4, self.num_samples)
        return np.exp(1j * (2 * np.pi * data / 4 + np.pi / 4))

    def generate_8psk(self):
        data = np.random.randint(0, 8, self.num_samples)
        return np.exp(1j * (2 * np.pi * data / 8))

    def generate_16qam(self):
        data = np.random.randint(0, 16, self.num_samples)
        mapping = [-3, -1, 1, 3]
        return np.array([mapping[d >> 2] for d in data]) + 1j * np.array([mapping[d & 3] for d in data])

    def generate_64qam(self):
        data = np.random.randint(0, 64, self.num_samples)
        mapping = [-7, -5, -3, -1, 1, 3, 5, 7]
        return np.array([mapping[d >> 3] for d in data]) + 1j * np.array([mapping[d & 7] for d in data])

    def generate_pam4(self):
        mapping = [-3, -1, 1, 3]
        return np.array([mapping[d] for d in np.random.randint(0, 4, self.num_samples)])

    def generate_gfsk(self):
        data = 2 * np.random.randint(0, 2, self.num_samples) - 1
        t = np.arange(self.num_samples)
        return np.exp(1j * np.cumsum(0.5 * np.pi * data / 8))

    def generate_cpfsk(self):
        data = 2 * np.random.randint(0, 2, self.num_samples) - 1
        return np.exp(1j * np.cumsum(0.5 * np.pi * data))

    # --- Analog Modulations ---
    def generate_am_dsb(self):
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t)
        carrier = np.cos(2 * np.pi * 50 * t)
        return (1 + message) * carrier

    def generate_am_ssb(self):
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t)
        # Using Hilbert transform-like shift for SSB
        message_hilbert = np.cos(2 * np.pi * 5 * t)
        carrier_i = np.cos(2 * np.pi * 50 * t)
        carrier_q = np.sin(2 * np.pi * 50 * t)
        return message * carrier_i - message_hilbert * carrier_q

    def generate_wbfm(self):
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t)
        integral_m = np.cumsum(message) / self.num_samples
        return np.cos(2 * np.pi * 50 * t + 10 * np.pi * integral_m)

if __name__ == "__main__":
    gen = AMCSignalGenerator()
    print("Signal Generator for all 11 modulations is ready.")