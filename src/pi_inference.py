import time
import socket
import struct
import numpy as np
from dataset import RadioMLDataset

# Network Configuration (Port reverted back to 5005 to match Raspberry Pi)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Loading dataset for ESP32 simulation...")
# Load dataset from data directory, matching original preprocessing parameters
ds = RadioMLDataset("data/RML2016.10a_dict.pkl", snr_min=-20, augment=False)

def compute_fft_magnitude(iq_np):
    I_ch = iq_np[0]
    Q_ch = iq_np[1]
    complex_sig = I_ch + 1j * Q_ch
    fft_mag = np.abs(np.fft.fft(complex_sig))
    fft_mag = np.fft.fftshift(fft_mag)
    std = fft_mag.std()
    if std > 1e-8:
        fft_mag = (fft_mag - fft_mag.mean()) / std
    return fft_mag.astype(np.float32)

print("Starting simulated ESP32 transmission stream (Infinite Loop)...")
try:
    while True: # Keep streaming forever, looping back to start of dataset
        for iq_tensor, label, snr in ds:
            # Convert Torch Tensor to stable NumPy Array immediately
            iq = iq_tensor.cpu().numpy() if hasattr(iq_tensor, "is_cuda") and iq_tensor.is_cuda else iq_tensor.numpy()
            
            # 1. Feature Extraction (Match model 5-channel requirements)
            I = iq[0]
            Q = iq[1]
            
            # Channel 3: Instantaneous Frequency
            phase = np.arctan2(Q, I)
            phase_unwrapped = np.unwrap(phase)
            inst_freq = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])
            if inst_freq.std() > 1e-8:
                inst_freq = (inst_freq - inst_freq.mean()) / inst_freq.std()
                
            # Channel 4: Amplitude Envelope
            envelope = np.sqrt(I**2 + Q**2)
            if envelope.std() > 1e-8:
                envelope = (envelope - envelope.mean()) / envelope.std()
                
            # Channel 5: FFT Magnitude
            fft_mag = compute_fft_magnitude(iq)
            
            # 2. Stack into (5, 128) Matrix
            matrix = np.vstack([I, Q, inst_freq, envelope, fft_mag]).astype(np.float32)
            
            # 3. Convert matrix to raw binary float bytes (5 * 128 * 4 = 2560 bytes)
            packet_data = matrix.tobytes()
            
            # 4. Stream over UDP to Raspberry Pi
            sock.sendto(packet_data, (UDP_IP, UDP_PORT))
            
            print(f"[ESP32 Sim] Sent packet | True SNR: {int(snr.item()):3d} dB | Class IDX: {label.item()}")
            time.sleep(0.5) # Stream delay interval (2 packets per second)
            
except KeyboardInterrupt:
    print("\nSimulation stopped.")