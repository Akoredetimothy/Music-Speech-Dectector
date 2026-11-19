# FFT Music/Speech Separation (Baseline)

STFT/band-separation baseline for exploring music + speech component separation. v1 uses simple frequency-band masking (no trained model).

## Structure

- `local/baseline_fft_separation.py` — local demo (no GPU/torch). Runs on a synthetic mix by default and saves spectrogram/recovery plots.
- `colab/01_fft_music_speech_demo.ipynb` — Colab demo notebook (synthetic mix → STFT → band separation → ISTFT → plots).
- `colab/music_speech_colab.py` — importable Colab helpers (preprocess, STFT/band separation, simple SNR proxy, demo).

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

```bash
pip install -r requirements.txt
```

## Usage (local)

```bash
python local/baseline_fft_separation.py --out ./out
```

If `.wav` files are present in `./clips`, the script reports the clip count and still writes the synthetic spectrogram demo to `--out`. For real-clip workflows, use/adapt the Colab notebook/module.

Outputs: `out/spectrogram_demo.png` (and related plots).

## Notes

- FFT/STFT baseline only; no Demucs/PANNs or model training in v1.
