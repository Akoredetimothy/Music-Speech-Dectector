#!/usr/bin/env python3
"""
Environment: LOCAL only.
Project: A - Music + Speech (FFT separation, v1 baseline)

Local code (run on your own machine):
- No large downloads, no Colab, no GPU/torch required.
- Deps: numpy, scipy, matplotlib (optional: librosa if installed).
- Input: put tiny WAV files (mono, 44.1 kHz, 10-20 s) in ./clips
  or run with no files for synthetic demo mode.
- Output: spectrogram PNGs + recovered tone sources (if synthetic).

What this file is NOT:
- Not the Colab notebook. That is in ../colab/01_fft_music_speech_demo.ipynb
  (server-side, larger-data capable). Do not run this file in Colab.
- Not a trained model. No Demucs/PANNs here. FFT/STFT baseline only.

Install (local):
  pip install numpy scipy matplotlib
"""

import argparse
import pathlib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

SR = 44100
NFFT = 2048
NOVERLAP = 1536

def synthetic_mix(sr=SR, dur=4.0):
    t = np.arange(int(sr * dur)) / sr
    s1 = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    s2 = 0.5 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    return s1, s2, (s1 + s2) * 0.5

def stft_mag(x):
    f, t, Z = signal.stft(x, fs=SR, nperseg=NFFT, noverlap=NOVERLAP)
    return f, t, Z

def band_separate(Z, f, f_lo1, f_hi1, f_lo2, f_hi2):
    mask1 = ((f >= f_lo1) & (f < f_hi1)).astype(float)[:, None]
    mask2 = ((f >= f_lo2) & (f < f_hi2)).astype(float)[:, None]
    return Z * mask1, Z * mask2

def plot_spec(ax, f, t, Z, title):
    mag = np.abs(Z)
    im = ax.pcolormesh(t, f, 20 * np.log10(mag + 1e-9),
                       shading="gouraud", cmap="magma", vmin=-60, vmax=0)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im

def save_demo_pngs(outdir: pathlib.Path):
    outdir.mkdir(parents=True, exist_ok=True)
    s1, s2, mix = synthetic_mix()
    f, t, Z = stft_mag(mix)
    Z1, Z2 = band_separate(Z, f, 150, 280, 260, 380)
    _, x1 = signal.istft(Z1, fs=SR, nperseg=NFFT, noverlap=NOVERLAP)
    _, x2 = signal.istft(Z2, fs=SR, nperseg=NFFT, noverlap=NOVERLAP)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plot_spec(axes[0, 0], f, t, Z, "Mix STFT")
    plot_spec(axes[0, 1], f, t, Z1, "Band 1 (220 Hz region)")
    plot_spec(axes[1, 0], f, t, Z2, "Band 2 (330 Hz region)")
    axes[1, 1].plot(np.linspace(0, 4, len(x1)), x1, label="rec1")
    axes[1, 1].plot(np.linspace(0, 4, len(x2)), x2, label="rec2")
    axes[1, 1].set_title("Recovered waveforms")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(outdir / "spectrogram_demo.png", dpi=130)

def main():
    ap = argparse.ArgumentParser(description="Local FFT separation demo.")
    ap.add_argument("--clips", default="./clips", help="Folder of .wav files (optional)")
    ap.add_argument("--out", default="./out", help="Output folder for PNGs")
    args = ap.parse_args()
    clips = pathlib.Path(args.clips)
    out = pathlib.Path(args.out)
    if not list(clips.glob("*.wav")):
        save_demo_pngs(out)
        print(f"No clips in {clips}, saved synthetic demo to {out}/spectrogram_demo.png")
    else:
        out.mkdir(parents=True, exist_ok=True)
        save_demo_pngs(out)
        print(f"Saved spectrograms for {len(list(clips.glob('*.wav')))} clips to {out}/")

if __name__ == "__main__":
    main()
