#!/usr/bin/env python3
"""
Environment: COLAB only (importable module, not local).
Project: A - Music + Speech - remaining tasks (steps 2-4)

Import in Colab:
  # Upload this file to Colab, or copy into notebook cell:
  from music_speech_colab import demo, evaluate_sdr, preprocess_clip, separate_bands, stft_mag

What this covers (remaining from plan):
  Step 2: tiny real slice via librosa example / upload hook (server-side, ~0 GB to user)
  Step 3: preprocess (trim, loudnorm -16 LUFS via ffmpeg if present, else RMS)
  Step 4: baseline FFT evaluation (spectrogram + simple band SNR proxy)
  Step 6 (conditional): Demucs inference stub - NOT imported by default (200+ MB, heavy)

No torch/GPU required for steps 2-4. Demucs left as stub.
"""
import pathlib, json
import numpy as np
from scipy import signal

SR = 44100
NFFT = 2048
NOVERLAP = 1536

def ensure_deps():
    try:
        import librosa
        return True
    except ImportError:
        return False

def get_demo_audio():
    """Return (s1, s2, mix) synthetic if librosa example fails. No download."""
    if ensure_deps():
        try:
            import librosa
            y, _ = librosa.load(librosa.ex("trumpet"), sr=SR, duration=4.0)
            y = y[:int(SR*4)]
            return y, np.zeros_like(y), y
        except Exception:
            pass
    t = np.arange(int(SR*4.0))/SR
    s1 = 0.6*np.sin(2*np.pi*220*t)
    s2 = 0.5*np.sin(2*np.pi*330*t)
    return s1, s2, (s1+s2)*0.5

def preprocess_clip(x, sr=SR, target_duration=20.0):
    """Trim/pad, RMS normalize. Loudnorm via ffmpeg is optional, done in notebook shell."""
    x = x - np.mean(x)
    m = np.max(np.abs(x)) or 1.0
    x = x / m * 0.8
    n = int(sr*target_duration)
    if len(x) > n:
        x = x[:n]
    elif len(x) < n:
        x = np.pad(x, (0, n-len(x)))
    return x

def stft_mag(x):
    f, t, Z = signal.stft(x, fs=SR, nperseg=NFFT, noverlap=NOVERLAP)
    return f, t, Z

def separate_bands(Z, f, bands):
    """bands = [(lo,hi),...] returns list of masked Z."""
    out = []
    for lo, hi in bands:
        mask = ((f >= lo) & (f < hi)).astype(float)[:, None]
        out.append(Z * mask)
    return out

def istft(Z):
    _, x = signal.istft(Z, fs=SR, nperseg=NFFT, noverlap=NOVERLAP)
    return x

def evaluate_sdr(reference, estimate):
    """Simple SNR proxy (dB) when band separation succeeds. Not a full SDR."""
    n = min(len(reference), len(estimate))
    r, e = reference[:n], estimate[:n]
    num = np.mean(r**2) + 1e-12
    den = np.mean((r-e)**2) + 1e-12
    return float(10*np.log10(num/den))

def demo(outdir="/tmp/music_out"):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    s1, s2, mix = get_demo_audio()
    mix = preprocess_clip(mix)
    f, t, Z = stft_mag(mix)
    Z1, Z2 = separate_bands(Z, f, [(150,280),(260,380)])
    x1, x2 = istft(Z1), istft(Z2)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3, figsize=(14,4))
    for a, Z_, ttl in zip(ax, [Z,Z1,Z2], ["Mix","Band1","Band2"]):
        a.pcolormesh(t, f, 20*np.log10(np.abs(Z_)+1e-9), shading="gouraud", cmap="magma", vmin=-60, vmax=0)
        a.set_ylim(0,4000); a.set_title(ttl)
    plt.tight_layout(); plt.savefig(f"{outdir}/spectrograms.png", dpi=130)
    result = {"sdr_band1": round(evaluate_sdr(s1, x1),2), "sdr_band2": round(evaluate_sdr(s2, x2),2)}
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    print(demo())
