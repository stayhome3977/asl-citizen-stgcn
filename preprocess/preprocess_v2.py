"""preprocess_v2.py - Tiền xử lý keypoint ASL theo khung chốt v2.0 (P1-P5).

ĐẦU VÀO (giữ nguyên layout raw hiện tại để bên K không phải đổi):
    raw: mảng (N, 107) keypoint thô của MỘT video, tọa độ MediaPipe trong [0,1],
    khung mất để NaN. Layout cột:
        0..41    tay phải 21 điểm (x,y)       (K4: phải đứng trước)
        42..83   tay trái 21 điểm (x,y)
        84..104  7 điểm pose (x,y,vis): nose, rsho, relb, rwri, lsho, lelb, lwri
        105,106  cờ tay phải/trái (KHÔNG dùng; có tay = tọa độ hữu hạn)
    W, H: độ phân giải video (để sửa méo tỉ lệ 4:3 / 16:9). fps: để chuẩn hóa ngưỡng.

ĐẦU RA: (feats (T,100) float32, mask (T,2) float32, info dict)
    feats cột:  0..41 tay phải theo cổ tay | 42..83 tay trái theo cổ tay |
                84..95 cánh tay [rsho,relb,rwri,lsho,lelb,lwri] x (x,y) theo vai |
                96,97 cỡ bàn tay phải/trái (chia độ rộng vai) | 98,99 mũi (x,y)
    mask: [thấy tay phải, thấy tay trái] BỞI MEDIAPIPE ở mỗi khung (1 = tay thật,
    0 = khung được nội suy/giữ giá trị hoặc không có tay).
    Mẫu không có tay nào: khối tay = 0 và cỡ = 0 (tay thật luôn có điểm khác 0
    sau khi trừ cổ tay nên không nhầm được).
Mọi tham số nằm ở phần CẤU HÌNH; các ngưỡng là GIÁ TRỊ KHỞI ĐẦU, phải chỉnh bằng
biểu đồ trên tập TRAIN (xem build_data_v2.py --plot-trim).
"""
from __future__ import annotations
import numpy as np

# ---------------- CẤU HÌNH ----------------
T_OUT = 64
RAW_DIM = 107
FEAT_DIM = 100
MAX_GAP = 1000           # nội suy mọi khoảng mất tay (thử nghiệm: tốt hơn giới hạn 8 khung)
SHOULDER_VIS = 0.2
MOTION_ABS = 0.012       # đã chọn bằng trim_check.png trên train (0.004 cắt quá nhẹ)
MOTION_REL = 0.05        # hoặc 5% đỉnh chuyển động của video
SMOOTH_WIN = 3
TRIM_PAD = 4
MIN_JOINTS = 4
MIN_SEG = 8              # D5: dưới 8 khung thì loại
LOW_HAND_RATIO = 0.5     # D5: mất tay quá nửa -> cờ LOW_HAND (để builder quyết định)
MIN_HAND_REL = 0.4       # cỡ bàn tay < 40% trung vị của video -> coi là hỏng
LOCAL_CLIP = 5.0         # chặn tọa độ tay theo cổ tay trong [-5, 5]
FALLBACK_SCALE = 0.25    # độ rộng vai giả định (tỉ lệ ảnh) khi không thấy vai

RH_SL, LH_SL, PS0 = slice(0, 42), slice(42, 84), 84
OUT = {"rh": slice(0, 42), "lh": slice(42, 84), "arm": slice(84, 96),
       "size": slice(96, 98), "nose": slice(98, 100)}
# thứ tự pose trong raw: 0 nose,1 rsho,2 relb,3 rwri,4 lsho,5 lelb,6 lwri
ARM_IDX = [1, 2, 3, 4, 5, 6]


# ---------------- HÀM PHỤ ----------------
def _split(raw, W, H):
    hr = raw[:, RH_SL].reshape(-1, 21, 2).copy()
    hl = raw[:, LH_SL].reshape(-1, 21, 2).copy()
    ps = raw[:, PS0:PS0 + 21].reshape(-1, 7, 3)
    pose, vis = ps[:, :, :2].copy(), ps[:, :, 2].copy()
    for a in (hr, hl, pose):          # sang đơn vị pixel để hết méo tỉ lệ
        a[..., 0] *= W
        a[..., 1] *= H
    return hr, hl, pose, vis


def _body_frame(pose, vis):
    """Tâm vai, độ rộng vai: TRUNG VỊ trên các khung có cả hai vai (ổn định, ít rung)."""
    ok = (np.isfinite(pose[:, [1, 4]]).all(axis=(1, 2))
          & (vis[:, 1] > SHOULDER_VIS) & (vis[:, 4] > SHOULDER_VIS))
    if not ok.any():
        return None
    r, l = pose[ok, 1], pose[ok, 4]
    c = np.median((r + l) / 2, axis=0)
    s = float(np.median(np.hypot(*(r - l).T)))
    return c[0], c[1], max(s, 1e-6), float(ok.mean())


def _fill_gaps(a, max_gap):
    """Nội suy tuyến tính khoảng mất <= max_gap; dài hơn thì để NaN.
    Trả (mảng, ok_sau_nội_suy, real) trong đó real = khung MediaPipe thật sự thấy tay."""
    a = a.copy()
    n = len(a)
    ok = np.isfinite(a).all(axis=1)
    a[~ok] = np.nan
    out_ok = ok.copy()
    if not ok.any():
        return a, out_ok, ok.copy()
    idx = np.flatnonzero(ok)
    for s, e in zip(idx[:-1], idx[1:]):
        gap = e - s - 1
        if 0 < gap <= max_gap:
            w = (np.arange(s + 1, e) - s) / (e - s)
            a[s + 1:e] = (1 - w)[:, None] * a[s] + w[:, None] * a[e]
            out_ok[s + 1:e] = True
    first, last = idx[0], idx[-1]
    if 0 < first <= max_gap:
        a[:first] = a[first]
        out_ok[:first] = True
    if 0 < n - 1 - last <= max_gap:
        a[last + 1:] = a[last]
        out_ok[last + 1:] = True
    return a, out_ok, ok


def _interp_cols(a):
    out = a.copy()
    x = np.arange(len(a))
    for j in range(a.shape[1]):
        g = np.isfinite(a[:, j])
        if g.any() and not g.all():
            out[~g, j] = np.interp(x[~g], x[g], a[g, j])
    return out


def _motion(pts):
    n = len(pts)
    d = np.full(n, np.nan)
    for t in range(1, n):
        both = np.isfinite(pts[t - 1]) & np.isfinite(pts[t])
        if both.sum() >= MIN_JOINTS:
            v = pts[t][both] - pts[t - 1][both]
            d[t] = np.mean(np.hypot(v[0::2], v[1::2]))
    return d


def _trim(hr, hl, pose, body, fps):
    """Trả (start, end, thr, motion_mượt). Không đòi có tay: chuyển động lấy từ
    cả bàn tay lẫn cánh tay nên MediaPipe bỏ sót tay không làm cắt nhầm."""
    n = len(hr)
    cx, cy, sc = body[:3] if body else (0.0, 0.0, FALLBACK_SCALE)
    pts = np.concatenate([hr.reshape(n, 42), hl.reshape(n, 42),
                          pose[:, 1:].reshape(n, 12)], axis=1).reshape(n, -1, 2)
    pts = ((pts - [cx, cy]) / sc).reshape(n, -1)
    m = _motion(pts) * (fps / 30.0)
    if not np.isfinite(m).any():
        return 0, n, float("nan"), m
    x = np.arange(n)
    bad = ~np.isfinite(m)
    m[bad] = np.interp(x[bad], x[~bad], m[~bad])
    sm = np.convolve(m, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
    thr = max(MOTION_ABS, MOTION_REL * float(sm.max()))
    act = sm > thr
    if not act.any():
        return 0, n, thr, sm
    first = int(np.argmax(act))
    last = int(n - 1 - np.argmax(act[::-1]))
    return max(0, first - TRIM_PAD), min(n, last + 1 + TRIM_PAD), thr, sm


def _clean_hand(h):
    """Khung có cỡ bàn tay quá nhỏ so với cỡ điển hình của chính video (MediaPipe làm
    méo) bị coi là mất tay, để không sinh giá trị cực đoan khi chia cho cỡ bàn tay."""
    pts = h.reshape(-1, 21, 2)
    hs = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=1)
    ok = np.isfinite(h).all(axis=1) & np.isfinite(hs) & (hs > 1e-6)
    if ok.any():
        ok &= hs > MIN_HAND_REL * np.median(hs[ok])
    h = h.copy()
    h[~ok] = np.nan
    return h


def _hand(h_f, ok, sc):
    pts = h_f.reshape(-1, 21, 2)
    hs = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=1)
    good = ok & np.isfinite(hs) & (hs > 1e-6)
    local = np.zeros((len(pts), 42))
    size = np.zeros(len(pts))
    local[good] = np.clip((pts[good] - pts[good, 0:1]) / hs[good, None, None],
                          -LOCAL_CLIP, LOCAL_CLIP).reshape(-1, 42)
    size[good] = hs[good] / sc
    return local, size, good


# ---------------- HÀM CHÍNH ----------------
def preprocess(raw, W=None, H=None, fps=30.0, T=T_OUT):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM or len(raw) == 0:
        raise ValueError(f"raw phải là (N,{RAW_DIM}), nhận {raw.shape}")
    W = float(W) if W else 1.0
    H = float(H) if H else 1.0
    n0 = len(raw)
    hr, hl, pose, vis = _split(raw, W, H)
    body = _body_frame(pose, vis)
    s, e, thr, _ = _trim(hr, hl, pose, body, fps)
    hr, hl, pose = hr[s:e], hl[s:e], pose[s:e]
    n = e - s
    cx, cy, sc = body[:3] if body else (0.5 * W, 0.5 * H, FALLBACK_SCALE * W)

    hr_f, ok_r, real_r = _fill_gaps(_clean_hand(hr.reshape(n, 42)), MAX_GAP)
    hl_f, ok_l, real_l = _fill_gaps(_clean_hand(hl.reshape(n, 42)), MAX_GAP)
    loc_r, size_r, good_r = _hand(hr_f, ok_r, sc)
    loc_l, size_l, good_l = _hand(hl_f, ok_l, sc)

    P = _interp_cols(pose.reshape(n, 14)).reshape(n, 7, 2)
    P = np.nan_to_num((P - [cx, cy]) / sc)

    feats = np.concatenate([loc_r, loc_l, P[:, ARM_IDX].reshape(n, 12),
                            np.stack([size_r, size_l], 1), P[:, 0]], axis=1)
    # mask = MediaPipe THẬT SỰ thấy tay ở khung đó (không tính khung nội suy/giữ giá trị)
    mask = np.stack([real_r, real_l], 1).astype(np.float32)

    idx = np.linspace(0, n - 1, T).round().astype(int)   # đều cho MỌI độ dài
    feats = feats[idx].astype(np.float32)
    mask = mask[idx]

    any_ratio = float((real_r | real_l).mean())
    if body is None:
        status = "NO_SHOULDER"
    elif n < MIN_SEG:
        status = "TOO_SHORT"
    elif any_ratio == 0:
        status = "NO_HAND"
    elif any_ratio < LOW_HAND_RATIO:
        status = "LOW_HAND"
    else:
        status = "OK"
    info = dict(n_frames_raw=n0, seg_start=s, seg_end=e, seg_len=n,
                trim_thr=thr, right_ratio=round(float(real_r.mean()), 3),
                left_ratio=round(float(real_l.mean()), 3),
                filled_ratio=round(float((good_r | good_l).mean() - any_ratio), 3),
                any_hand_ratio=round(any_ratio, 3),
                shoulder_ok_ratio=round(body[3], 3) if body else 0.0,
                status=status)
    return feats, mask, info


def debug_trim(raw, W=None, H=None, fps=30.0):
    """Để vẽ biểu đồ chọn ngưỡng: trả (motion mượt, ngưỡng, start, end)."""
    raw = np.asarray(raw, dtype=np.float64)
    hr, hl, pose, vis = _split(raw, float(W or 1), float(H or 1))
    s, e, thr, sm = _trim(hr, hl, pose, _body_frame(pose, vis), fps)
    return sm, thr, s, e


def flip_x(X):
    """Lật ngang cho augmentation (X: (...,100)); đổi chỗ phải<->trái.
    Với mask dùng mask[..., ::-1]."""
    X = np.asarray(X)
    Y = X.copy()
    lead = X.shape[:-1]
    rh = X[..., OUT["rh"]].reshape(*lead, 21, 2).copy()
    lh = X[..., OUT["lh"]].reshape(*lead, 21, 2).copy()
    rh[..., 0] *= -1
    lh[..., 0] *= -1
    Y[..., OUT["rh"]] = lh.reshape(*lead, 42)
    Y[..., OUT["lh"]] = rh.reshape(*lead, 42)
    arm = X[..., OUT["arm"]].reshape(*lead, 6, 2).copy()
    arm[..., 0] *= -1
    Y[..., OUT["arm"]] = arm[..., [3, 4, 5, 0, 1, 2], :].reshape(*lead, 12)
    Y[..., 96], Y[..., 97] = X[..., 97], X[..., 96]
    nose = X[..., OUT["nose"]].copy()
    nose[..., 0] *= -1
    Y[..., OUT["nose"]] = nose
    return Y


# ---------------- TỰ KIỂM TRA ----------------
def _synthetic(W=640, H=480, shift=(0, 0), k=1.0, seed=0):
    rng = np.random.default_rng(seed)
    N = 100
    tmpl = rng.normal(0, 1, (21, 2)) * 12
    tmpl[0] = 0
    tmpl[9] = [0, -40]                       # cỡ bàn tay = 40 px
    t = np.arange(N)
    move = ((t >= 20) & (t < 80)).astype(float)
    rw = np.stack([250 + 60 * np.sin((t - 20) / 8) * move,
                   380 - 50 * np.sin((t - 20) / 5) * move], 1)
    raw = np.full((N, RAW_DIM), np.nan)
    hr = rw[:, None, :] + tmpl[None]
    hr[40:45] = np.nan                       # khoảng mất NGẮN (5)
    hr[66:] = np.nan                         # khoảng mất DÀI ở cuối
    pose = np.zeros((N, 7, 3))
    pose[:, :, 2] = 1.0
    pose[:, 0, :2] = [320, 200]
    pose[:, 1, :2] = [260, 300]
    pose[:, 2, :2] = [240, 380]
    pose[:, 3, :2] = rw
    pose[:, 4, :2] = [380, 300]
    pose[:, 5, :2] = [400, 380]
    pose[:, 6, :2] = [410, 300]
    hr = hr * k + np.array(shift)
    pose[:, :, :2] = pose[:, :, :2] * k + np.array(shift)
    raw[:, RH_SL] = (hr / [W, H]).reshape(N, 42)
    ps = pose.copy()
    ps[:, :, 0] /= W
    ps[:, :, 1] /= H
    raw[:, PS0:PS0 + 21] = ps.reshape(N, 21)
    return raw   # tay trái vắng suốt video


if __name__ == "__main__":
    W, H = 640, 480
    f, m, info = preprocess(_synthetic(), W, H)
    print(info)
    assert f.shape == (T_OUT, FEAT_DIM) and f.dtype == np.float32
    assert np.isfinite(f).all() and m.shape == (T_OUT, 2)
    assert 14 <= info["seg_start"] <= 20 and 80 <= info["seg_end"] <= 88, "cắt đoạn lệch"
    assert m[:, 1].sum() == 0 and (f[:, OUT["lh"]] == 0).all(), "tay trái phải vắng"
    assert (f[:, 97] == 0).all()
    assert 0 < m[:, 0].sum() < T_OUT, "khoảng nội suy phải có mask = 0"
    assert np.abs(f[m[:, 0] == 0][:, OUT["rh"]]).sum() > 0, "MAX_GAP lớn: khung nội suy phải có giá trị"
    assert info["filled_ratio"] > 0
    g0 = MAX_GAP
    globals()["MAX_GAP"] = 8                       # giới hạn nội suy: khoảng dài phải về 0
    f8, m8, _ = preprocess(_synthetic(), W, H)
    globals()["MAX_GAP"] = g0
    assert (f8[m8[:, 0] == 0][:, OUT["rh"]] == 0).any(), "gap=8 phải để 0 ở khoảng mất dài"
    # bất biến khi người ký đứng chỗ khác / xa gần khác (thay đổi tuyến tính theo pixel)
    f2, m2, i2 = preprocess(_synthetic(shift=(40, -25), k=1.3), W, H)
    assert i2["seg_start"] == info["seg_start"] and i2["seg_end"] == info["seg_end"]
    assert np.allclose(f, f2, atol=1e-4), "không bất biến dịch/co giãn"
    # lật ngang hai lần = ban đầu
    assert np.allclose(flip_x(flip_x(f)), f)
    # gọi hai lần cho cùng kết quả (train vs demo)
    assert np.array_equal(f, preprocess(_synthetic(), W, H)[0])
    # đoạn ngắn: chỉ số lấy mẫu phải phủ đều, không đệm đuôi
    idx = np.linspace(0, 19, T_OUT).round().astype(int)
    assert idx[0] == 0 and idx[-1] == 19 and (np.diff(idx) >= 0).all()
    print("TẤT CẢ KIỂM TRA ĐẠT")
