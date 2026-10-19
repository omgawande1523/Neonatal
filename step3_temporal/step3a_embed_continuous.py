# =====================================================================
# EdgeSeiz-Net  STEP 3a: embed the CONTINUOUS recordings of all 24 patients
# Reads every EDF once, cuts the same 4 s / 2 s-step windows as window_index.csv,
# runs the frozen clean CNN (v1clean_raw.keras) and saves, per window:
#   16-d embedding (GAP layer), probability, label, start time.
# This gives (a) continuous VALIDATION recordings for honest FA/h thresholds and
# (b) the input sequences for the Step 3b temporal (GRU) head. No training here.
# Resumable: complete patients are skipped; patients with files missing are redone.
# EDFs missing from Drive are downloaded straight from PhysioNet (public, no login).
# =====================================================================
import subprocess, sys
try:
    import pyedflib
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyedflib'], check=True)
    import pyedflib
import os, re, time, glob, shutil, zipfile, json
import numpy as np, pandas as pd

# ---------------- CONFIG ----------------
DRIVE_ROOT   = None
FS, WIN_SEC, STEP_SEC = 256, 4.0, 2.0
COMMON_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
EXCLUDE_FILES = {'chb12_27.edf', 'chb12_28.edf', 'chb12_29.edf'}   # CS2 montage
TEST_PATIENTS = {'chb01', 'chb09', 'chb17', 'chb19'}
FLUSH_EVERY = 6            # force a Drive flush after this many patients
LOCAL_EDF = '/content/edf_tmp'
PHYSIONET = ['https://physionet.org/files/chbmit/1.0.0',              # same public data your notebook
             'https://physionet-open.s3.amazonaws.com/chbmit/1.0.0']  # first synced via aws s3
IN_COLAB = os.path.exists('/content')

# ---------------- 1. MOUNT (retry + force_remount) ----------------
def mount_drive(tries=5):
    if not IN_COLAB:
        return
    from google.colab import drive
    for i in range(tries):
        try:
            drive.mount('/content/drive', force_remount=(i > 0))
            os.listdir('/content/drive/MyDrive')
            return
        except Exception as e:
            print(f'[mount] attempt {i+1} failed: {e}')
            time.sleep(5 * (i + 1))
    raise RuntimeError('Drive mount failed repeatedly')

mount_drive()

# A folder in "Shared with me" is only visible to Colab after you right-click it in Drive ->
# Organize -> Add shortcut -> My Drive. It then appears under MyDrive or .shortcut-targets-by-id.
def find_root():
    cands = [DRIVE_ROOT] if DRIVE_ROOT else []
    cands += ['/content/drive/MyDrive/chbmit']
    for pat in ('/content/drive/MyDrive/*', '/content/drive/MyDrive/*/*',
                '/content/drive/.shortcut-targets-by-id/*/*', '/content/drive/.shortcut-targets-by-id/*/*/*',
                '/content/drive/Shareddrives/*/*', '/content/drive/Shareddrives/*/*/*'):
        cands += sorted(glob.glob(pat))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'shards')) and os.path.isdir(os.path.join(c, 'test_shards_full')):
            return c
    raise FileNotFoundError(
        'Could not find a folder containing shards/ and test_shards_full/. If the data is in '
        '"Shared with me": open drive.google.com -> Shared with me -> right-click the folder -> '
        'Organize -> Add shortcut -> My Drive, then re-run this cell.')

DRIVE_ROOT = find_root() if IN_COLAB or not DRIVE_ROOT else DRIVE_ROOT
TRAIN_SRC, TEST_SRC = f'{DRIVE_ROOT}/shards', f'{DRIVE_ROOT}/test_shards_full'
print(f'Data folder: {DRIVE_ROOT}')

# Outputs go next to the data if you can write there, else to your own Drive.
OUT_DIR = f'{DRIVE_ROOT}/step1_spectrogram'
try:
    os.makedirs(OUT_DIR, exist_ok=True)
    open(f'{OUT_DIR}/.write_test', 'w').write('ok'); os.remove(f'{OUT_DIR}/.write_test')
except OSError:
    OUT_DIR = '/content/drive/MyDrive/edgeseiz_step1' if IN_COLAB else OUT_DIR + '_local'
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Data folder is read-only for you (shared as viewer); writing outputs to your own Drive.')
print(f'Outputs: {OUT_DIR}')
# v4b: the earlier v4_* files were trained/early-stopped on test patients (shards/ holds all 24) -> do not use them
MODEL_PATH = f'{OUT_DIR}/v1clean_raw.keras'
PRED_PATH  = f'{OUT_DIR}/v1clean_test_predictions.npz'

# ---------------- 2. COPY + VALIDATE (real read of every array) ----------------
def npz_ok(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]               # forces full decompression + CRC check
                _ = a.sum() if a.dtype.kind in 'fiub' else a.shape
        return True
    except Exception:
        return False

def copy_and_validate(src_dir, dst_dir, tries=4):
    os.makedirs(dst_dir, exist_ok=True)
    for t in range(tries):
        try:
            names = sorted(f for f in os.listdir(src_dir) if f.endswith('.npz'))
            break
        except OSError as e:
            print(f'[list] {e}; remounting'); mount_drive()
    good, bad = [], []
    t0 = time.time()
    for i, n in enumerate(names):
        src, dst = os.path.join(src_dir, n), os.path.join(dst_dir, n)
        ok = os.path.exists(dst) and npz_ok(dst)
        for t in range(tries):
            if ok:
                break
            try:
                shutil.copyfile(src, dst + '.part')
                os.replace(dst + '.part', dst)
                ok = npz_ok(dst)
                if not ok:
                    print(f'[copy] {n} copied but unreadable (attempt {t+1})')
            except OSError as e:
                print(f'[copy] {n}: {e}; remounting (attempt {t+1})')
                mount_drive()
        (good if ok else bad).append(dst)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(names)} shards ok ({time.time()-t0:.0f}s)')
    print(f'{src_dir}: {len(good)} valid, {len(bad)} BAD -> {[os.path.basename(b) for b in bad]}')
    return good, bad

def npz_ok(path):   # same real-read check, without float16 overflow warnings
    try:
        with np.load(path, allow_pickle=False) as z, np.errstate(over='ignore'):
            for k in z.files:
                a = z[k]; _ = a.astype(np.float64).sum() if a.dtype.kind in 'fiub' else a.shape
        return True
    except Exception:
        return False

import tensorflow as tf, urllib.request
EMB_DIR    = os.path.join(OUT_DIR, 'emb')
BACKUP_DIR = ('/content/drive/MyDrive/edgeseiz_outputs' if IN_COLAB else OUT_DIR + '_backup') + '/emb'
for d_ in (EMB_DIR, BACKUP_DIR, LOCAL_EDF):
    os.makedirs(d_, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
gap = [l for l in model.layers if 'global_average_pooling' in l.name][-1]
embedder = tf.keras.Model(model.inputs, [gap.output, model.outputs[0]])
CH_FIRST = tuple(model.input_shape[1:3]) == (18, 1024)
print(f'Model {MODEL_PATH}: embedding size {gap.output.shape[-1]}, input {model.input_shape}')

ann = pd.read_csv(os.path.join(DRIVE_ROOT, 'master_annotations.csv'))
ann['has'] = ann['has_seizure'].astype(str).str.lower().isin(['1', 'true', 'yes'])

def copy_edf(src, dst, tries=4):
    for t in range(tries):
        try:
            shutil.copyfile(src, dst + '.part'); os.replace(dst + '.part', dst)
            with pyedflib.EdfReader(dst) as _: pass
            return True
        except Exception as e:
            print(f'   [edf] {os.path.basename(src)}: {type(e).__name__} {e}; retry {t+1}')
            if isinstance(e, OSError): mount_drive()
    return False

def download_edf(subject, fname, dst, tries=4):
    for t in range(tries):
        url = f'{PHYSIONET[t % len(PHYSIONET)]}/{subject}/{fname}'
        try:
            urllib.request.urlretrieve(url, dst + '.part'); os.replace(dst + '.part', dst)
            with pyedflib.EdfReader(dst) as _: pass
            print(f'   {fname}: not in Drive -> downloaded from PhysioNet')
            return True
        except Exception as e:
            print(f'   [download] {fname}: {type(e).__name__} {e}; retry {t+1}'); time.sleep(5 * (t + 1))
    return False

def embed_file(subject, fname):
    src = os.path.join(DRIVE_ROOT, subject, fname); dst = os.path.join(LOCAL_EDF, fname)
    if not os.path.exists(src):
        mount_drive()                                   # FUSE listings can go stale; look once more
    if not (os.path.exists(src) and copy_edf(src, dst)):
        if not download_edf(subject, fname, dst):
            return None, 'unreadable'
    f = pyedflib.EdfReader(dst)
    labels = f.getSignalLabels(); idx = {l: i for i, l in enumerate(labels)}
    if not all(c in idx for c in COMMON_CHANNELS):
        f.close(); os.remove(dst); return None, 'missing channels'
    sig = np.stack([f.readSignal(idx[c]).astype(np.float32) for c in COMMON_CHANNELS])
    dur = f.file_duration; f.close(); os.remove(dst)
    starts = []
    t = 0.0
    while t + WIN_SEC <= dur:                     # identical to window_index.csv construction
        starts.append(t); t += STEP_SEC
    starts = np.array(starts, np.float32)
    sz = [(r.seizure_start_sec, r.seizure_end_sec) for r in ann[(ann.subject == subject) & (ann.file == fname) & ann.has].itertuples()]
    y = np.array([int(any(s < e2 and s + WIN_SEC > s2 for s2, e2 in sz)) for s in starts], np.int8)
    E, P = [], []
    for b in range(0, len(starts), 2048):
        idx0 = (starts[b:b + 2048] * FS).astype(int)
        X = np.stack([sig[:, i:i + int(WIN_SEC * FS)] for i in idx0])      # (n, 18, 1024)
        e, p = embedder.predict(X if CH_FIRST else np.transpose(X, (0, 2, 1)), batch_size=512, verbose=0)
        E.append(e.astype(np.float16)); P.append(p.ravel().astype(np.float32))
    return dict(emb=np.concatenate(E), prob=np.concatenate(P), y=y, start=starts), 'ok'

def expected_files(p):
    return set(ann[ann.subject == p].file) - EXCLUDE_FILES

def patient_done(p):
    """Saved AND complete: every expected EDF is either embedded or permanently unusable (missing channels)."""
    for d_ in (EMB_DIR, BACKUP_DIR):
        f_ = os.path.join(d_, f'{p}.npz')
        if os.path.exists(f_) and npz_ok(f_):
            with np.load(f_) as z:
                have = set(map(str, z['file_names'])) | set(map(str, z['skipped'] if 'skipped' in z.files else []))
            if have >= expected_files(p):
                return f_
    return None

patients = sorted(ann.subject.unique())
skipped, since_flush, t0 = [], 0, time.time()
for p in patients:
    if patient_done(p):
        print(f'{p}: already complete, skipping'); continue
    files = sorted(expected_files(p))
    parts, names = [], []
    for fn in files:
        r, why = embed_file(p, fn)
        if r is None:
            skipped.append((p, fn, why)); print(f'   skip {fn}: {why}'); continue
        parts.append(r); names.append(fn)
    if not parts:
        continue
    out = dict(emb=np.concatenate([r['emb'] for r in parts]), prob=np.concatenate([r['prob'] for r in parts]),
               y=np.concatenate([r['y'] for r in parts]), start=np.concatenate([r['start'] for r in parts]),
               file_id=np.concatenate([np.full(len(r['y']), i, np.int16) for i, r in enumerate(parts)]),
               file_names=np.array(names),
               skipped=np.array([fn for (pp_, fn, why) in skipped if pp_ == p and why == 'missing channels'], dtype=str))
    local = f'/content/{p}_emb.npz' if IN_COLAB else os.path.join(OUT_DIR, f'.{p}_emb.npz')
    np.savez_compressed(local, **out)
    for d_ in (EMB_DIR, BACKUP_DIR):
        shutil.copyfile(local, os.path.join(d_, f'{p}.npz'))
    os.remove(local)
    n = len(out['y'])
    print(f'{p}: {len(names)} files, {n:,} windows ({n*STEP_SEC/3600:.1f} h), {int(out["y"].sum())} seizure windows '
          f'| {(time.time()-t0)/60:.0f} min elapsed')
    since_flush += 1
    if IN_COLAB and since_flush >= FLUSH_EVERY:
        from google.colab import drive
        drive.flush_and_unmount(); time.sleep(5); mount_drive(); since_flush = 0

if IN_COLAB:
    from google.colab import drive
    print('Final flush to Drive...'); drive.flush_and_unmount(); time.sleep(5); mount_drive()

# ---------------- CHECKS ----------------
print(f'\nSkipped files ({len(skipped)}):', skipped)
tot_w = tot_s = 0
missing = [p for p in patients if not patient_done(p)]
for p in patients:
    f_ = patient_done(p)
    if f_:
        with np.load(f_) as z: tot_w += len(z['y']); tot_s += int(z['y'].sum())
print(f'Complete patients: {len(patients) - len(missing)}/{len(patients)} | incomplete or missing: {missing}')
print(f'Total windows {tot_w:,} (window_index.csv had 1,751,856) | seizure windows {tot_s:,} (had 6,105)')
wi_path = os.path.join(DRIVE_ROOT, 'window_index.csv')
if os.path.exists(wi_path):
    wi = pd.read_csv(wi_path, usecols=['subject', 'file', 'label'])
    wi = wi[~wi.file.isin(EXCLUDE_FILES)].groupby('subject').agg(n=('label', 'size'), s=('label', 'sum'))
    bad = []
    for p in patients:
        f_ = patient_done(p)
        if f_ and p in wi.index:
            with np.load(f_) as z:
                if (len(z['y']), int(z['y'].sum())) != (int(wi.loc[p, 'n']), int(wi.loc[p, 's'])):
                    bad.append((p, len(z['y']), int(wi.loc[p, 'n']), int(z['y'].sum()), int(wi.loc[p, 's'])))
    print('Per-patient windows/seizure windows vs window_index.csv:',
          'ALL MATCH' if not bad else f'MISMATCH (patient, ours, index, our sz, index sz): {bad}')

# The 4 test patients must reproduce the Step 1c predictions -> proves the EDF pipeline matches the shards.
pp = os.path.join(OUT_DIR, 'v1clean_test_predictions.npz')
if os.path.exists(pp):
    ref = np.load(pp); diffs = []
    for p in sorted(TEST_PATIENTS):
        f_ = patient_done(p)
        if not f_: continue
        z = np.load(f_)
        for i, fn in enumerate(z['file_names']):
            sh = f'{p}_{str(fn).replace(".edf", "")}.npz'
            j = np.where(ref['shard_names'] == sh)[0]
            if len(j) == 0: continue
            rp = ref['prob'][ref['shard'] == j[0]]; mp = z['prob'][z['file_id'] == i]
            if len(rp) != len(mp):
                print(f'   [check] {sh}: {len(mp)} windows vs {len(rp)} in shard'); continue
            diffs.append(np.abs(rp - mp).max())
    if diffs:
        print(f'Consistency vs Step 1c test predictions: {len(diffs)} files, max |diff| {max(diffs):.2e} '
              f'({"MATCH" if max(diffs) < 1e-3 else "MISMATCH - tell Claude"})')
print('\nDONE. Paste everything from "Model" down (the per-patient lines and the checks).')
