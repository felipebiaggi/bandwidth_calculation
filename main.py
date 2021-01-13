import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile

def bandwidth(wave_path):
    fs, data = wavfile.read(wave_path)

    # if input has more than 1 channel, only first will be used
    if len(data.shape) > 1:
        print("Audio is not mono! First channel will be used.")
        data = data[:, 0]

    # compute how many blocks there is in the input signal
    audio_duration = len(data) / fs  # in seconds
    block_size_in_samples = round(audio_duration * fs)

    first_sample = 0
    last_sample = block_size_in_samples
    first_sample = 0
    last_sample = block_size_in_samples
    if last_sample > len(data):
        last_sample = len(data)
    block_samples = data[first_sample:last_sample]

    # The pre-emphasis filter
    num = [1, -0.95]
    den = 1
    pre_emphasis_signal = lfilter(num, den, block_samples)

    # FFT of the entire signal
    X = np.fft.fft(pre_emphasis_signal)

    # X is an even signal, so we discard its second half
    X = X[0 : int(len(X) / 2 + 1)]

    # X, in dB
    XdB = 20 * np.log10(np.abs(X))

    # angular frequency
    w = np.linspace(0, np.pi, len(X))

    # Polynomial envelope
    p = np.polyfit(w, XdB, 7)
    polynomial_envelope = np.polyval(p, w)

    max_idx = np.argmax(polynomial_envelope)
    min_idx = np.argmin(polynomial_envelope[max_idx:]) + max_idx

    HdB_max = polynomial_envelope[max_idx]
    HdB_min = polynomial_envelope[min_idx]
    HdB_cut = np.amax([HdB_max - 25, HdB_min])

    cut_idx = np.argmin(np.abs(polynomial_envelope[max_idx:] - HdB_cut)) + max_idx

    f_Hz_envelope = (w / np.pi) * (fs / 2)

    bandwidth = f_Hz_envelope[cut_idx]

    conventional_freqs = [4000, 8000, 11025, 16000, 22500, 32000, 44100, 48000, 96000]

    idx = np.argmin(np.abs(conventional_freqs - bandwidth))
    bandwidth = conventional_freqs[idx]

    return bandwidth
