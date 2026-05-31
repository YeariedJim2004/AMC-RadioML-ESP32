import socket
import numpy as np
import struct

# Network Configurations
GNURADIO_IP = "127.0.0.1"
GNURADIO_PORT = 5000  # Matches GNU Radio UDP Sink

RPI_IP = "127.0.0.1"
RPI_PORT = 5001       # Routes to local Raspberry Pi simulator

# Initialize sockets
sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_in.bind((GNURADIO_IP, GNURADIO_PORT))

sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"ESP32 Simulator active. Listening to GNU Radio on port {GNURADIO_PORT}...")

def moving_average_denoise(signal, window=3):
    return np.convolve(signal, np.ones(window)/window, mode='same')

while True:
    # 128 Complex samples = 128 * 8 bytes = 1024 bytes buffer
    data, addr = sock_in.recvfrom(1024)
    if len(data) < 1024:
        continue
        
    # Decode binary stream into IQ floating arrays
    raw_floats = np.frombuffer(data, dtype=np.float32)
    I = raw_floats[0::2]
    Q = raw_floats[1::2]
    
    if len(I) != 128 or len(Q) != 128:
        continue

    # ESP32 DSP Layer: Moving average noise reduction
    I_clean = moving_average_denoise(I)
    Q_clean = moving_average_denoise(Q)
    
    # ESP32 DSP Layer: Signal envelope calculation
    envelope = np.sqrt(I_clean**2 + Q_clean**2)
    
    # Pack processed signals into binary payload (3 * 128 floats = 384 floats)
    payload = struct.pack(f'{128}f{128}f{128}f', *I_clean, *Q_clean, *envelope)
    sock_out.sendto(payload, (RPI_IP, RPI_PORT))