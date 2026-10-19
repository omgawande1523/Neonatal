# =====================================================================
# EdgeSeiz-Net  STEP 1c: retrain V1 (raw input) on the CLEAN patient split
# Finds your saved V1 model in Drive, re-initialises the same architecture,
# trains it without test patients, and scores old vs clean V1 with the V4 metrics.
# One self-contained Colab recovery cell. Safe to re-run after a crash:
#   - re-mounts Drive, re-copies only missing/broken shards
#   - skips training if the model checkpoint already exists in Drive
#   - saves test predictions to Drive for Step 2 (event-level metrics)
# =====================================================================
import os, re, time, glob, shutil, zipfile, json
import numpy as np

# ---------------- CONFIG ----------------
DRIVE_ROOT   = None   # None = auto-find the chbmit folder (own Drive, shortcut, or 'Shared with me')
LOCAL_TRAIN  = '/content/shards'
LOCAL_TEST   = '/content/test_shards_full'
V1_MODEL_PATH = None   # e.g. '/content/drive/MyDrive/chbmit/v1_model.keras' -> re-scores V1 with the SAME metric code
RETRAIN      = False   # True = ignore existing checkpoint and train again

FS, WIN_SEC, STEP_SEC = 256, 4.0, 2.0                    # 50% overlap
NPERSEG, NOVERLAP = 64, 32
BATCH, EPOCHS, PATIENCE = 64, 40, 6
CLASS_WEIGHT = {0: 1.0, 1: 5.0}                         # same as V1 for comparability
TEST_PATIENTS = {'chb01', 'chb09', 'chb17', 'chb19'}    # locked seed-42 split; NEVER trained or validated on
N_VAL_PATIENTS = 3                                      # patient-level val split from the 20 TRAIN patients
SEED = 42
SMOOTH_K = [1, 5]                                       # causal trailing mean over k windows
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

train_files, train_bad = copy_and_validate(TRAIN_SRC, LOCAL_TRAIN)
test_files,  test_bad  = copy_and_validate(TEST_SRC,  LOCAL_TEST)
assert not test_bad, 'Test shards are broken; fix before reporting any number.'

# ---------------- 3. DETECT SHARD LAYOUT ----------------
with np.load(train_files[0]) as z:
    print('Shard keys:', {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
    X_KEY = next(k for k in z.files if z[k].ndim == 3 and set(z[k].shape[1:]) == {18, 1024})
    n0 = z[X_KEY].shape[0]
    Y_KEY = next(k for k in z.files if k != X_KEY and z[k].ndim == 1 and z[k].shape[0] == n0
                 and z[k].dtype.kind in 'iubf' and set(np.unique(z[k])) <= {0, 1})
    CHANNELS_LAST_RAW = z[X_KEY].shape[1] == 1024
print(f'Using X={X_KEY!r}, y={Y_KEY!r}, raw layout {"(N,1024,18)" if CHANNELS_LAST_RAW else "(N,18,1024)"}')

def load_shard(path):
    with np.load(path) as z:
        x = z[X_KEY].astype(np.float32)
        y = z[Y_KEY].astype(np.float32)
        meta = {k: z[k] for k in z.files if k not in (X_KEY, Y_KEY) and z[k].ndim == 1 and len(z[k]) == len(y)}
    if CHANNELS_LAST_RAW:
        x = np.transpose(x, (0, 2, 1))
    return x, y, meta                                   # x: (N, 18, 1024)

# ---------------- 4. SPECTROGRAM ----------------
from scipy.signal import spectrogram
def to_spec(x):
    """(N,18,1024) raw -> (N,33,31,18) log-power, z-scored per window per channel over the TF plane."""
    _, _, S = spectrogram(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, axis=-1)   # (N,18,33,31)
    S = np.log10(S + 1e-10).astype(np.float32)
    mu = S.mean(axis=(2, 3), keepdims=True)
    sd = S.std(axis=(2, 3), keepdims=True) + 1e-6
    return np.transpose((S - mu) / sd, (0, 2, 3, 1)).astype(np.float32)

_t = time.time(); _x, _, _ = load_shard(train_files[0]); SPEC_SHAPE = to_spec(_x).shape[1:]
print(f'Spectrogram shape {SPEC_SHAPE}; {len(_x)/(time.time()-_t):.0f} windows/s incl. load (on-the-fly)')

# ---------------- 5. PATIENT-LEVEL VAL SPLIT (from TRAIN only) ----------------
def patient_of(path, meta=None):
    m = re.search(r'chb\d{2}', os.path.basename(path))
    return m.group(0) if m else None

from collections import Counter
print('shards/ patients:', dict(sorted(Counter(patient_of(f) for f in train_files).items())))
print('test_shards_full/ patients:', dict(sorted(Counter(patient_of(f) for f in test_files).items())))
assert all(patient_of(f) for f in train_files + test_files), \
    'Some shard names carry no chbXX id, so test-patient leakage cannot be ruled out. Stop and tell Claude.'
assert {patient_of(f) for f in test_files} <= TEST_PATIENTS, 'test_shards_full/ contains a non-test patient'
leaked = [f for f in train_files if patient_of(f) in TEST_PATIENTS]
if leaked:
    print(f'!! {len(leaked)} shards in shards/ belong to TEST patients -> excluded from training and validation')
train_files = [f for f in train_files if patient_of(f) not in TEST_PATIENTS]

pats = sorted({patient_of(f) for f in train_files})
assert not set(pats) & TEST_PATIENTS
rng = np.random.RandomState(SEED)
val_pats = {str(v) for v in rng.choice(pats, N_VAL_PATIENTS, replace=False)}
val_files = [f for f in train_files if patient_of(f) in val_pats]
trn_files = [f for f in train_files if patient_of(f) not in val_pats]
print(f'Train patients ({len(pats) - len(val_pats)}): {sorted(set(pats) - val_pats)}')
print(f'Val patients (held out of training, used for early stopping only): {sorted(val_pats)}')

def count(files):
    n = p = 0
    for f in files:
        with np.load(f) as z:
            y = z[Y_KEY]; n += len(y); p += int(y.sum())
    return n, p
n_trn, p_trn = count(trn_files); n_val, p_val = count(val_files)
print(f'train windows {n_trn} ({p_trn} seizure) | val windows {n_val} ({p_val} seizure)')

# ---------------- 6. FIND THE ORIGINAL V1 MODEL ----------------
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)
V1_PARAMS = 5375

def find_v1():
    if V1_MODEL_PATH:
        return V1_MODEL_PATH, tf.keras.models.load_model(V1_MODEL_PATH, compile=False)
    cands = []
    for root in {DRIVE_ROOT, '/content/drive/MyDrive'}:
        for pat in ('*.keras', '*.h5', '*/*.keras', '*/*.h5', '*/*/*.keras', '*/*/*.h5'):
            cands += glob.glob(os.path.join(root, pat))
    found = None
    for c in sorted(set(cands)):
        if 'step1_spectrogram' in c:
            continue
        try:
            m = tf.keras.models.load_model(c, compile=False)
        except Exception as e:
            print(f'  [skip] {c}: {type(e).__name__}'); continue
        print(f'  {c}: {m.count_params():,} params, input {m.input_shape}')
        if m.count_params() == V1_PARAMS and found is None:
            found = (c, m)
    return found if found else (None, None)

print('Searching Drive for saved models...')
v1_path, v1_orig = find_v1()
if v1_orig is not None:
    print(f'Using V1 = {v1_path}')
    v1_orig.summary()
else:
    print(f'!! No saved model with {V1_PARAMS:,} params found. Training a V1-LIKE stand-in instead; '
          'set V1_MODEL_PATH at the top if your V1 file is elsewhere.')

def standin_v1():
    inp = layers.Input((1024, 18)); x = inp
    for f, k in ((16, 7), (16, 7), (16, 5)):
        x = layers.Conv1D(f, k, strides=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(0.3)(x)
    return tf.keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='v1_like_standin')

# ---------------- 7. CLEAN RETRAIN (same architecture, fresh weights) ----------------
if os.path.exists(MODEL_PATH) and not RETRAIN:
    print(f'Loading existing clean model {MODEL_PATH} (set RETRAIN=True to retrain)')
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = tf.keras.models.clone_model(v1_orig) if v1_orig is not None else standin_v1()
model_is_new = not (os.path.exists(MODEL_PATH) and not RETRAIN)
RAW_CH_FIRST = tuple(model.input_shape[1:3]) == (18, 1024)
RAW_SHAPE = (18, 1024) if RAW_CH_FIRST else (1024, 18)
print(f'Raw input layout {RAW_SHAPE}')

def raw_in(x):
    return x if RAW_CH_FIRST else np.transpose(x, (0, 2, 1))

def make_gen(files, shuffle):
    def gen():
        order = np.random.permutation(len(files)) if shuffle else range(len(files))
        for i in order:
            x, y, _ = load_shard(files[i]); x = raw_in(x)
            j = np.random.permutation(len(y)) if shuffle else range(len(y))
            for k in j:
                yield x[k], y[k]
    return gen

if model_is_new:
    sig = (tf.TensorSpec(RAW_SHAPE, tf.float32), tf.TensorSpec((), tf.float32))
    ds_trn = (tf.data.Dataset.from_generator(make_gen(trn_files, True), output_signature=sig)
              .shuffle(4096).batch(BATCH).repeat().prefetch(tf.data.AUTOTUNE))
    ds_val = (tf.data.Dataset.from_generator(make_gen(val_files, False), output_signature=sig)
              .apply(tf.data.experimental.assert_cardinality(n_val)).batch(256).prefetch(tf.data.AUTOTUNE))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')])
    cbs = [tf.keras.callbacks.EarlyStopping('val_auc', mode='max', patience=PATIENCE, restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, 'val_auc', mode='max', save_best_only=True),
           tf.keras.callbacks.ReduceLROnPlateau('val_auc', mode='max', factor=0.5, patience=3),
           tf.keras.callbacks.CSVLogger(f'{OUT_DIR}/v1clean_train_log.csv', append=True)]
    model.fit(ds_trn, steps_per_epoch=n_trn // BATCH, validation_data=ds_val,
              epochs=EPOCHS, class_weight=CLASS_WEIGHT, callbacks=cbs, verbose=2)
    model.save(MODEL_PATH)
print(f'Params: {model.count_params():,}')

# ---------------- 8. PREDICT ON FULL HELD-OUT TEST SET ----------------
def predict_all(mdl):
    ch_first = tuple(mdl.input_shape[1:3]) == (18, 1024)
    P, Y, SH = [], [], []
    for si, f in enumerate(test_files):
        x, y, _ = load_shard(f)
        x = x if ch_first else np.transpose(x, (0, 2, 1))
        P.append(mdl.predict(x, batch_size=512, verbose=0).ravel()); Y.append(y)
        SH.append(np.full(len(y), si, np.int32))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(SH)

p, y, shard = predict_all(model)
np.savez_compressed(PRED_PATH, prob=p, y=y, shard=shard,
                    shard_names=np.array([os.path.basename(f) for f in test_files]))
print(f'Saved test predictions -> {PRED_PATH}')

# ---------------- 8b. MAKE SURE BOTH FILES REALLY REACH DRIVE ----------------
# Last time the files existed in the running Colab but never synced to Drive.
# So: keep a second copy in your own Drive, force a flush, remount, and re-read both.
BACKUP_DIR = '/content/drive/MyDrive/edgeseiz_outputs' if IN_COLAB else OUT_DIR + '_backup'
os.makedirs(BACKUP_DIR, exist_ok=True)
for p_ in (MODEL_PATH, PRED_PATH):
    shutil.copyfile(p_, os.path.join(BACKUP_DIR, os.path.basename(p_)))
if IN_COLAB:
    from google.colab import drive
    print('Flushing writes to Google Drive (can take a minute)...')
    drive.flush_and_unmount()
    time.sleep(5)
    mount_drive()
persist_ok = False
for p_ in (MODEL_PATH, PRED_PATH):
    for loc in (p_, os.path.join(BACKUP_DIR, os.path.basename(p_))):
        try:
            if loc.endswith('.npz'):
                ok = npz_ok(loc)
            else:
                tf.keras.models.load_model(loc); ok = True
        except Exception:
            ok = False
        persist_ok |= ok and loc.endswith('.keras')
        print(f'  {"OK     " if ok else "MISSING"} {loc}')
if not persist_ok:
    print('!! The model did not reach Drive. Do NOT close this runtime: tell Claude.')

# ---------------- 9. METRICS ----------------
from sklearn.metrics import roc_auc_score, average_precision_score
HOURS_TRUE   = len(y) * STEP_SEC / 3600     # actual recording time covered (50% overlap)
HOURS_NAIVE  = len(y) * WIN_SEC / 3600      # windows x 4 s: double-counts the overlap
print(f'\nTest: {len(y):,} windows, {int(y.sum())} seizure | '
      f'recording time {HOURS_TRUE:.1f} h (windows x 2 s); windows x 4 s would give {HOURS_NAIVE:.1f} h')

def smooth(p, shard, k):
    if k == 1: return p
    out = np.empty_like(p)
    for s in np.unique(shard):                  # causal trailing mean, never crosses shard boundary
        m = shard == s; c = np.cumsum(np.insert(p[m], 0, 0.0))
        i = np.arange(1, m.sum() + 1); lo = np.maximum(0, i - k)
        out[m] = (c[i] - c[lo]) / (i - lo)
    return out

def alarms(pred, y, shard):
    """Count false alarms as runs of consecutive positive windows containing no seizure window."""
    fa = 0
    for s in np.unique(shard):
        m = shard == s; pr, yy = pred[m], y[m]
        d = np.diff(np.concatenate([[0], pr.astype(np.int8), [0]]))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            fa += int(yy[a:b].sum() == 0)
    return fa

def report(p, y, shard, name):
    print(f'\n===== {name} =====')
    for k in SMOOTH_K:
        ps = smooth(p, shard, k)
        print(f'-- smoothing k={k}: AUROC {roc_auc_score(y, ps):.3f} | AUPRC {average_precision_score(y, ps):.4f} '
              f'(chance {y.mean():.4f})')
        print(f'   {"thr":>6} {"recall":>7} {"FPwin/h(true)":>14} {"FPwin/h(x4s)":>13} {"alarms/h(true)":>15} {"precision":>9}')
        o = np.argsort(-ps); tp_c = np.cumsum(y[o] == 1); fp_c = np.cumsum(y[o] == 0)
        shown = set()
        for target in (0.95, 0.9, 0.841, 0.8, 0.7, 0.606, 0.5, 0.3):
            i = int(np.searchsorted(tp_c, np.ceil(target * y.sum())))   # fewest windows reaching target recall
            if i >= len(o): continue
            t = ps[o[i]]
            if t in shown: continue
            shown.add(t)
            pr = ps >= t; tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
            fa = alarms(pr, y, shard)
            print(f'   {t:6.4f} {tp/y.sum():7.3f} {fp/HOURS_TRUE:14.1f} {fp/HOURS_NAIVE:13.1f} {fa/HOURS_TRUE:15.2f} {tp/max(1,tp+fp):9.4f}')
        pr = ps >= 0.5; tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
        print(f'   @0.5 (pre-specified): recall {tp/y.sum():.3f}, FPwin/h(true) {fp/HOURS_TRUE:.1f}, '
              f'alarms/h {alarms(pr, y, shard)/HOURS_TRUE:.2f}')


label = 'V1 architecture' if v1_orig is not None else 'V1-LIKE STAND-IN (not your V1)'
report(p, y, shard, f'{label}, CLEAN retrain, test patients excluded ({model.count_params():,} params)')
if v1_orig is not None:
    p0, _, _ = predict_all(v1_orig)
    report(p0, y, shard, f'ORIGINAL V1 weights, same metric code ({v1_orig.count_params():,} params)')
print('\nDONE. Paste everything from "Searching Drive" down.')
