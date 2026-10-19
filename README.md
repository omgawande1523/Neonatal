# EdgeSeiz-Net

Lightweight, edge-deployable (ESP32 / Raspberry Pi class) EEG seizure detector, trained and evaluated patient-independently on the CHB-MIT Scalp EEG Database. Full project brief: [docs/PROJECT_BRIEF.txt](docs/PROJECT_BRIEF.txt).

Data and trained models are not in this repo. They live on Google Drive (`/content/drive/MyDrive/chbmit`), and each script is a self-contained Colab cell that reads from there.

## Setup

- 18 bipolar channels selected by name, 4 s windows (18 x 1024 at 256 Hz), 50% overlap. chb12_27/28/29 excluded (different montage).
- Patient split (seed 42): test chb01, chb09, chb17, chb19 (17 seizure events). Validation chb02, chb20, chb22. Train the other 17.
- Evaluation only on `test_shards_full/` (natural class imbalance, 286,731 windows, 159.3 h of recording).
- `shards/` on Drive contains all 24 patients, so training code filters the 4 test patients out by filename.

## Pipeline

| Step | Script | Status |
|---|---|---|
| 1 Spectrogram SepConv2D baseline | `step1_spectrogram/step1_spectrogram_cell.py` | Done, negative (AUROC 0.824) |
| 1c Clean raw-input retrain (current benchmark) | `step1_spectrogram/step1c_v1_clean_retrain.py` | Done |
| 2 / 2b Event-level evaluation | `step2_event_level/step2_event_level.py`, `step2b_event_threshold.py` | Superseded (alarm counting merged long positive runs) |
| 2c Event-level with 60 s refractory alarm counting | `step2_event_level/step2c_event_fixed_fa.py` | Next to run |
| 3a Embeddings on continuous recordings for a temporal head | `step3_temporal/step3a_embed_continuous.py` | Drafted |
| 4 INT8 quantization, pruning, ESP32 benchmarks | | Planned |
| 5 Grad-CAM explanations | | Planned |

`notebooks/CHBMIT.ipynb` holds the earlier V1 to V3 experiments. Those used test patients for validation, so their numbers are not reported.

## Current results (step 1c, 5,153 params, raw input)

- Window level: AUROC 0.938 (k=1 smoothing) / 0.955 (k=5); AUPRC 0.50 / 0.56.
- Operating points (k=5): recall 0.70 at 4.6 alarms/h, 0.61 at 1.6/h, 0.50 at 0.33/h.
- Event level: 17/17 test seizures detected, latency 4 to 20 s. Thresholds chosen on the 3 validation patients do not transfer well to test, which steps 2c and 3 address.

## Next for the paper

Leave-one-patient-out over all 24 patients, then TUSZ for seizure-type classification once credentials arrive.
