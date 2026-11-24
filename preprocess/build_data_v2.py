"""build_data_v2.py - đóng gói data_v2.npz từ keypoint thô (cho phần P).

1) Dữ liệu giả cho Dũng/T Hoàng (M7), chạy được NGAY:
     python build_data_v2.py --fake 120 --out data_v2_fake.npz

2) Biểu đồ chọn ngưỡng cắt (chỉ nhìn TRAIN):
     python build_data_v2.py --make-manifest raw_keypoints extract_log.csv   # tạo manifest.csv
     python build_data_v2.py --manifest manifest.csv --plot-trim 12

3) Đóng gói thật:
     python build_data_v2.py --manifest manifest.csv --out data_v2.npz

manifest.csv gồm cột: sample_id, participant_id, split, label, raw_path
  và tùy chọn: width, height, fps.   raw_path là file .npy (N,107) của 1 video.
Nhãn lấy từ cột label của CSV gốc (KHÔNG parse tên file). Thứ tự nhãn A-Z (D6).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess_v2 import preprocess, debug_trim, T_OUT, FEAT_DIM

DROP = {"NO_SHOULDER", "TOO_SHORT", "NO_HAND"}   # D5: loại hẳn; LOW_HAND chỉ gắn cờ


# Định dạng raw của nhóm: .npz gồm hands (N,2,21,3) [0=phải,1=trái; x,y,z] và
# pose (N,7,3) theo thứ tự gốc MediaPipe: nose, Lsho, Rsho, Lelb, Relb, Lwri, Rwri,
# kênh thứ 3 của pose là z (KHÔNG phải visibility). Đổi sang layout 107 cột của preprocess_v2:
POSE_MAP = [0, 2, 4, 6, 1, 3, 5]   # -> nose, rsho, relb, rwri, lsho, lelb, lwri


def npz_to_raw107(z):
    h, p = z["hands"], z["pose"][:, POSE_MAP]
    n = len(h)
    raw = np.full((n, 107), np.nan)
    raw[:, 0:42] = h[:, 0, :, :2].reshape(n, 42)
    raw[:, 42:84] = h[:, 1, :, :2].reshape(n, 42)
    ps = np.zeros((n, 7, 3))
    ps[:, :, :2] = p[:, :, :2]
    ps[:, :, 2] = np.isfinite(p[:, :, 0]).astype(float)   # không có visibility -> 1 nếu có điểm
    raw[:, 84:105] = ps.reshape(n, 21)
    raw[:, 105:107] = 0
    return raw


def make_manifest(raw_dir, extract_log, out="manifest.csv"):
    """Ghép extract_log.csv với file .npz trong raw_dir/{split}/{gloss}/."""
    ex = pd.read_csv(extract_log)
    rows = []
    for r in ex.itertuples():
        stem = r.video_file.rsplit(".mp4", 1)[0]
        p = Path(raw_dir) / r.split / r.gloss / f"{stem}.npy.npz"
        if not p.exists():
            p = p.with_name(f"{stem}.npz")
        rows.append(dict(sample_id=stem, participant_id=r.participant, split=r.split,
                         label=r.gloss, raw_path=str(p), exists=p.exists()))
    m = pd.DataFrame(rows)
    print("Khớp file:", int(m.exists.sum()), "/", len(m))
    m[m.exists].drop(columns="exists").to_csv(out, index=False)


def load_row(r):
    if str(r.raw_path).endswith(".npz"):
        raw = npz_to_raw107(np.load(r.raw_path))
    else:
        raw = np.load(r.raw_path)
    W = getattr(r, "width", None)
    H = getattr(r, "height", None)
    fps = getattr(r, "fps", None)
    W = None if pd.isna(W) else W
    H = None if pd.isna(H) else H
    fps = 30.0 if fps is None or pd.isna(fps) else float(fps)
    return raw, W, H, fps


def build(manifest, out, signs_out="signs.json"):
    man = pd.read_csv(manifest)
    labels = sorted(man.label.unique())
    lid = {l: i for i, l in enumerate(labels)}
    json.dump(lid, open(signs_out, "w"), ensure_ascii=False, indent=1)
    rows, X, M, log = [], [], [], []
    for r in man.itertuples():
        raw, W, H, fps = load_row(r)
        f, m, info = preprocess(raw, W, H, fps)
        info.update(sample_id=r.sample_id, participant_id=r.participant_id,
                    split=r.split, label=r.label)
        if info["status"] in DROP:
            log.append(info)
            continue
        rows.append(info); X.append(f); M.append(m)
    meta = pd.DataFrame(rows)
    X, M = np.stack(X), np.stack(M)
    y = meta.label.map(lid).values.astype(np.int64)
    d = {}
    for sp in ("train", "val", "test"):
        k = (meta.split == sp).values
        d[f"X_{sp}"], d[f"M_{sp}"], d[f"y_{sp}"] = X[k], M[k], y[k]
        d[f"signer_{sp}"] = meta.participant_id[k].values.astype(str)
        d[f"id_{sp}"] = meta.sample_id[k].values.astype(str)
    np.savez_compressed(out, **d)
    meta.to_csv(Path(out).with_suffix(".meta.csv"), index=False)
    pd.DataFrame(log).to_csv(Path(out).with_suffix(".dropped.csv"), index=False)
    check(d, meta, len(labels))
    print("Đã loại:", len(log), "| gắn cờ LOW_HAND:", int((meta.status == "LOW_HAND").sum()))


def check(d, meta, n_cls):
    print("== Kiểm tra ==")
    for sp in ("train", "val", "test"):
        X = d[f"X_{sp}"]
        assert X.shape[1:] == (T_OUT, FEAT_DIM) and X.dtype == np.float32
        assert np.isfinite(X).all(), f"NaN trong {sp}"
        assert d[f"y_{sp}"].dtype == np.int64
    s = {sp: set(d[f"signer_{sp}"]) for sp in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (s[a] & s[b]), f"trùng người ký {a}-{b}: {s[a] & s[b]}"
    ct = pd.crosstab(meta.label, meta.split).reindex(columns=["train", "val", "test"], fill_value=0)
    bad = ct[(ct.train < 12) | (ct.val < 3) | (ct.test < 8)]
    print(f"Số mẫu train/val/test: {[len(d[f'y_{sp}']) for sp in ('train','val','test')]}")
    print(f"Từ không đạt D2 (12/3/8): {len(bad)}/{n_cls}")
    if len(bad):
        print(bad.to_string())
    print("Phân bố status:", meta.status.value_counts().to_dict())


def fake(n, out, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    cut = {"train": int(n * .6), "val": int(n * .2)}
    cut["test"] = n - cut["train"] - cut["val"]
    for sp, k in cut.items():
        d[f"X_{sp}"] = rng.normal(0, .5, (k, T_OUT, FEAT_DIM)).astype(np.float32)
        d[f"M_{sp}"] = (rng.random((k, T_OUT, 2)) > .3).astype(np.float32)
        d[f"y_{sp}"] = rng.integers(0, 20, k).astype(np.int64)
        d[f"signer_{sp}"] = np.array([f"{sp[0]}{i % 4}" for i in range(k)])
        d[f"id_{sp}"] = np.array([f"fake_{sp}_{i}" for i in range(k)])
    np.savez_compressed(out, **d)
    print("Đã tạo dữ liệu giả:", out, {sp: d[f"X_{sp}"].shape for sp in cut})


def plot_trim(manifest, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    man = pd.read_csv(manifest)
    man = man[man.split == "train"].sample(min(n, (man.split == "train").sum()), random_state=0)
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(len(man) / cols)), cols, figsize=(14, 3 * np.ceil(len(man) / cols)))
    for ax, r in zip(np.ravel(axs), man.itertuples()):
        raw, W, H, fps = load_row(r)
        sm, thr, s, e = debug_trim(raw, W, H, fps)
        ax.plot(sm); ax.axhline(thr, color="r", ls="--")
        ax.axvspan(s, e, alpha=.2, color="g"); ax.set_title(f"{r.label} ({len(raw)}kh)", fontsize=8)
    plt.tight_layout(); plt.savefig("trim_check.png", dpi=110)
    print("Đã lưu trim_check.png (xanh = đoạn giữ, đỏ = ngưỡng)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--out", default="data_v2.npz")
    ap.add_argument("--fake", type=int); ap.add_argument("--plot-trim", type=int)
    ap.add_argument("--make-manifest", nargs=2, metavar=("RAW_DIR", "EXTRACT_LOG"))
    a = ap.parse_args()
    if a.make_manifest:
        make_manifest(*a.make_manifest)
    elif a.fake:
        fake(a.fake, a.out)
    elif a.plot_trim:
        plot_trim(a.manifest, a.plot_trim)
    elif a.manifest:
        build(a.manifest, a.out)
    else:
        ap.print_help(); sys.exit(1)
