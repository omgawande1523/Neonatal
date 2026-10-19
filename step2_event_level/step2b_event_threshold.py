# =====================================================================
# EdgeSeiz-Net  STEP 2b: event-level metrics with EVENT-BASED threshold selection on val
# Uses the saved clean-V1 test predictions (no retraining). Steps:
#   a) checks windows are in time order by matching label runs to master_annotations.csv
#   b) picks thresholds on the VALIDATION patients only (never on test), by asking
#      'highest threshold that still catches all / >=90% of val seizures'
#   c) reports event sensitivity, FA/h, FA/24h, detection latency, per patient
# One self-contained Colab recovery cell; safe to re-run.
# =====================================================================
import os, re, time, glob, shutil, zipfile, json
import numpy as np, pandas as pd

# ---------------- CONFIG ----------------
DRIVE_ROOT   = None   # None = auto-find the chbmit folder
LOCAL_TRAIN  = '/content/shards'
PRED_NAME    = 'v1clean_test_predictions.npz'   # from Step 1c
FS, WIN_SEC, STEP_SEC = 256, 4.0, 2.0
TEST_PATIENTS = {'chb01', 'chb09', 'chb17', 'chb19'}
N_VAL_PATIENTS = 3
SEED = 42
SMOOTH_K = [1, 5, 10]          # causal trailing mean over k windows (k=10 -> 22 s of context)
MERGE_GAP_S = 30               # alarms closer than this are merged into one alarm
VAL_EVENT_TARGETS = [1.0, 0.9]   # fraction of VAL seizure events the threshold must still catch
VAL_FPWIN_TARGETS = [20, 5]      # alt rule: val false-positive windows/h (estimated from the 15:1 negatives)
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

def copy_and_validate(src_dir, dst_dir, tries=4, keep=lambda n: True):
    os.makedirs(dst_dir, exist_ok=True)
    for t in range(tries):
        try:
            names = sorted(f for f in os.listdir(src_dir) if f.endswith('.npz') and keep(f))
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

import tensorflow as tf
PRED_PATH = os.path.join(OUT_DIR, PRED_NAME)
if not os.path.exists(PRED_PATH):
    PRED_PATH = os.path.join(DRIVE_ROOT, 'step1_spectrogram', PRED_NAME)
assert os.path.exists(PRED_PATH) and os.path.exists(MODEL_PATH), f'Run Step 1c first (missing {PRED_PATH} or {MODEL_PATH})'
d = np.load(PRED_PATH)
prob, y, shard, shard_names = d['prob'], d['y'].astype(int), d['shard'], d['shard_names']
print(f'Loaded {PRED_PATH}: {len(y):,} windows, {y.sum()} seizure windows, {len(shard_names)} files')
MERGE_GAP = int(MERGE_GAP_S / STEP_SEC)

# ---------------- a) ORDER / ALIGNMENT CHECK ----------------
ann = pd.read_csv(os.path.join(DRIVE_ROOT, 'master_annotations.csv'))
ann['stem'] = ann['file'].astype(str).str.replace('.edf', '', regex=False).str.strip()
seiz = ann[ann['has_seizure'].astype(str).str.lower().isin(['1', 'true', 'yes']) & ann['seizure_start_sec'].notna()]

def stem_of(name):   # 'chb01_chb01_03.npz' -> 'chb01_03'
    s = str(name)[:-4]
    return s.split('_', 1)[1] if s.count('_') >= 2 else s

events = []          # (shard_idx, start_win, end_win_exclusive, patient)
bad_align = 0
for si, nm in enumerate(shard_names):
    m = np.where(shard == si)[0]; yy = y[m]
    dd = np.diff(np.concatenate([[0], yy, [0]]))
    runs = list(zip(np.where(dd == 1)[0], np.where(dd == -1)[0]))
    exp = seiz[seiz['stem'] == stem_of(nm)].sort_values('seizure_start_sec')
    for (a, b) in runs:
        events.append((si, m[a], m[0] + b, str(nm)[:5]))
    if len(runs) != len(exp):
        bad_align += 1
        print(f'  [align] {nm}: {len(runs)} label runs vs {len(exp)} annotated seizures')
    else:
        for (a, b), (_, r) in zip(runs, exp.iterrows()):
            if abs(a * STEP_SEC - r['seizure_start_sec']) > 6:
                bad_align += 1
                print(f'  [align] {nm}: run starts {a*STEP_SEC:.0f}s, annotation {r["seizure_start_sec"]:.0f}s')
print(f'Events in test set: {len(events)} | annotated: {len(seiz[seiz["stem"].isin([stem_of(n) for n in shard_names])])} '
      f'| alignment problems: {bad_align}')
if bad_align:
    print('!! Windows may not be in time order or not start at 0 s: event metrics below are approximate.')
ev_pat = np.array([e[3] for e in events])
hours_pat = {p: (np.char.startswith(shard_names.astype(str), p)[shard]).sum() * STEP_SEC / 3600
             for p in sorted(TEST_PATIENTS)}
HOURS = len(y) * STEP_SEC / 3600

# ---------------- b) VALIDATION THRESHOLDS ----------------
pats = sorted({re.search(r'chb\d{2}', f).group(0) for f in os.listdir(TRAIN_SRC) if f.endswith('.npz')} - TEST_PATIENTS)
val_pats = {str(v) for v in np.random.RandomState(SEED).choice(pats, N_VAL_PATIENTS, replace=False)}
print(f'Val patients: {sorted(val_pats)}')
val_files, _ = copy_and_validate(TRAIN_SRC, LOCAL_TRAIN, keep=lambda n: n[:5] in val_pats)
with np.load(val_files[0]) as z:
    X_KEY = next(k for k in z.files if z[k].ndim == 3); Y_KEY = next(k for k in z.files if z[k].ndim == 1)
model = tf.keras.models.load_model(MODEL_PATH)
ch_first = tuple(model.input_shape[1:3]) == (18, 1024)
vp, vy, vf = [], [], []
for fi, f in enumerate(val_files):
    with np.load(f) as z:
        x = z[X_KEY].astype(np.float32); vy.append(z[Y_KEY].astype(int))
    if x.shape[1] == 1024: x = np.transpose(x, (0, 2, 1))
    vp.append(model.predict(x if ch_first else np.transpose(x, (0, 2, 1)), batch_size=512, verbose=0).ravel())
    vf.append(np.full(len(vy[-1]), fi))
vp, vy, vf = np.concatenate(vp), np.concatenate(vy), np.concatenate(vf)

# Val shards are rebalanced, but every seizure window is kept, so each seizure is a run of 1s in its file.
val_events, vbad = [], 0
for fi, f in enumerate(val_files):
    m = np.where(vf == fi)[0]
    dd = np.diff(np.concatenate([[0], vy[m], [0]]))
    runs = [(m[0] + a, m[0] + b) for a, b in zip(np.where(dd == 1)[0], np.where(dd == -1)[0])]
    n_exp = len(seiz[seiz['stem'] == stem_of(os.path.basename(f))])
    if len(runs) != n_exp:
        vbad += 1; print(f'  [val align] {os.path.basename(f)}: {len(runs)} runs vs {n_exp} annotated')
    val_events += runs
print(f'Val seizure events: {len(val_events)} | alignment problems: {vbad}')

def run_peak(scores, k):   # max of causal trailing mean inside one seizure run
    c = np.cumsum(np.insert(scores, 0, 0.0)); i = np.arange(1, len(scores) + 1); lo = np.maximum(0, i - k)
    return float(((c[i] - c[lo]) / (i - lo)).max())

neg = vp[vy == 0]
val_thr_k = {}
for k in SMOOTH_K:
    peaks = np.sort([run_peak(vp[a:b], k) for a, b in val_events])[::-1]
    d = {}
    for tgt in VAL_EVENT_TARGETS:
        n_need = int(np.ceil(tgt * len(peaks)))
        d[f'val events>={tgt:.0%}'] = float(peaks[n_need - 1]) - 1e-6
    for fpw in VAL_FPWIN_TARGETS:
        d[f'val FPwin/h<={fpw}'] = float(np.quantile(neg, 1 - fpw / (3600 / STEP_SEC)))
    val_thr_k[k] = d
    print(f'k={k} val-chosen thresholds:', {n: round(t, 4) for n, t in d.items()})

# ---------------- c) EVENT METRICS ----------------
def smooth(p, k):
    if k == 1: return p
    out = np.empty_like(p)
    for s in np.unique(shard):
        m = np.where(shard == s)[0]; c = np.cumsum(np.insert(p[m], 0, 0.0))
        i = np.arange(1, len(m) + 1); lo = np.maximum(0, i - k)
        out[m] = (c[i] - c[lo]) / (i - lo)
    return out

def alarm_runs(pred):
    """Alarm intervals [a, b) in global window index, merged within MERGE_GAP, never across files."""
    runs = []
    for s in np.unique(shard):
        m = np.where(shard == s)[0]; pr = pred[m].astype(np.int8)
        dd = np.diff(np.concatenate([[0], pr, [0]]))
        a_s, b_s = np.where(dd == 1)[0], np.where(dd == -1)[0]
        cur = None
        for a, b in zip(a_s, b_s):
            if cur and a - cur[1] <= MERGE_GAP: cur[1] = b
            else:
                if cur: runs.append((m[0] + cur[0], m[0] + cur[1]))
                cur = [a, b]
        if cur: runs.append((m[0] + cur[0], m[0] + cur[1]))
    return runs

def evaluate(ps, t):
    pred = ps >= t
    runs = alarm_runs(pred)
    hit, lat = np.zeros(len(events), bool), np.full(len(events), np.nan)
    fa_pat = {p: 0 for p in TEST_PATIENTS}
    for a, b in runs:
        overl = [j for j, (_, s, e, _) in enumerate(events) if a < e and b > s]
        if not overl:
            fa_pat[str(shard_names[shard[a]])[:5]] += 1
    for j, (_, s, e, _) in enumerate(events):
        w = np.where(pred[s:e])[0]
        if len(w): hit[j] = True; lat[j] = w[0] * STEP_SEC + WIN_SEC   # alarm time minus seizure onset
    fa = sum(fa_pat.values())
    return dict(sens=hit.mean(), n_hit=int(hit.sum()), fa_h=fa / HOURS, lat=np.nanmedian(lat) if hit.any() else np.nan,
                hit=hit, fa_pat=fa_pat)

print(f'\nTest: {len(events)} seizure events, {HOURS:.1f} h. Alarms merged within {MERGE_GAP_S}s. '
      f'Each event = {100/len(events):.1f}% sensitivity, so treat differences of 1 event as noise.')
for k in SMOOTH_K:
    ps = smooth(prob, k)
    print(f'\n===== smoothing k={k} =====')
    print('  VAL-CHOSEN thresholds (honest operating points):')
    print(f'   {"rule":>18} {"thr":>7} {"events":>8} {"sens":>6} {"FA/h":>7} {"FA/24h":>8} {"med.latency":>12}')
    for r, t in val_thr_k[k].items():
        e = evaluate(ps, t)
        print(f'   {r:>18} {t:7.4f} {e["n_hit"]:>3}/{len(events):<4} {e["sens"]:6.3f} {e["fa_h"]:7.2f} {e["fa_h"]*24:8.1f} {e["lat"]:10.0f} s')
    print('  Full curve (test-swept; for the ROC-style figure only, NOT an operating point):')
    print(f'   {"thr":>7} {"sens":>6} {"FA/h":>7} {"FA/24h":>8}')
    best = {}
    for t in np.unique(np.quantile(ps, 1 - np.logspace(-5.5, -0.7, 60))):
        e = evaluate(ps, t)
        best.setdefault(e['n_hit'], (e['fa_h'], t))
        if e['fa_h'] < best[e['n_hit']][0]: best[e['n_hit']] = (e['fa_h'], t)
    for n in sorted(best, reverse=True)[:8]:
        fa, t = best[n]
        print(f'   {t:7.4f} {n/len(events):6.3f} {fa:7.2f} {fa*24:8.1f}')

# per-patient breakdown at the val-chosen 0.8-recall threshold, k=5
k, r = 5, 'val events>=90%'
e = evaluate(smooth(prob, k), val_thr_k[k][r])
print(f'\nPer patient (k={k}, rule "{r}", threshold {val_thr_k[k][r]:.4f}):')
for p in sorted(TEST_PATIENTS):
    sel = ev_pat == p
    print(f'  {p}: events {int(e["hit"][sel].sum())}/{int(sel.sum())}, FA/h {e["fa_pat"][p]/max(hours_pat[p], 1e-9):.2f} '
          f'({hours_pat[p]:.1f} h)')
print('\nDONE. Paste everything from "Loaded" down.')
