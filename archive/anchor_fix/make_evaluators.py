"""Make the anchor-fix copies of the four evaluators behind the HESS paper's January to August 2026 results.

Farid asked on 2026-09-24 to fix the forecast anchor and rerun everything. The originals are not edited. Each
patched copy sits next to its original, so its sibling imports still resolve, and is named *_anchorfix.py.

Fix 1, all four evaluators: the forecast anchor.
  Before: z0 = z_win[-1]. When the last reading is masked (for example by the 24 h flatline screen), Z holds 0 there,
  which is the gauge's training mean, so the forecast started from the gauge mean instead of the current stage.
  After: z0 is the last valid stage of the 72 h history (or of the repaired history when postprocessing repaired it);
  with no valid reading in the window, the raw stage at t0; with neither, 0 as before. When the last reading is
  valid, the new anchor equals the old one.
Fix 2, the LSTM evaluator only: its flatline screen used a centred 96-step window, which looks 12 h ahead and
  differs from the trailing window of the three System B evaluators. It now uses the same trailing window.

Every substitution is counted, and the script stops if a target is missing or repeated.
"""
import os

EXPERIMENTS = "project/Experiments/"
EVALUATORS = [
    "archive/anchor_fix/hindcast_before_fix.py",
    EXPERIMENTS + "SYSTEM_B_DROPEDGE_DROPNODE_20260915/scripts/evaluate_system_b_dropout.py",
    "archive/anchor_fix/lstm_inference_before_fix.py",
    "archive/anchor_fix/simulated_operational_forecast_before_fix.py",
]

ANCHOR_FUNCTION = '''

def last_valid_anchor(stage_window, p):
    """Anchor fix (2026-09-24): start the forecast from the last valid stage of the 72 h history.

    The model predicts a change from its anchor. The old anchor, z_win[-1], is 0 (the gauge's training mean) when the
    last reading is masked, so a frozen sensor made the forecast start from the gauge mean. With no valid reading in
    the window, the raw stage at t0 is used, and with neither, the old value 0.
    """
    valid = np.isfinite(stage_window)
    anchor = np.zeros(N, np.float32)
    has_valid = valid.any(axis=0)
    last_row = stage_window.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
    last_value = stage_window[last_row, np.arange(N)]
    anchor[has_valid] = (last_value[has_valid] - mu[has_valid]) / sd[has_valid]
    raw_now = S_RAW[p]
    use_raw = (~has_valid) & np.isfinite(raw_now)
    anchor[use_raw] = (raw_now[use_raw] - mu[use_raw]) / sd[use_raw]
    return anchor.astype(np.float32)
'''

OLD_ANCHOR = "        z0[r] = z_win[-1]\n"
NEW_ANCHOR = (
    "        anchor_window = s_hist if (POST and fixed) else S[start:p + 1]\n"
    "        z0[r] = last_valid_anchor(anchor_window, p)\n"
)
Z_LINE = "Z = np.nan_to_num((S - mu[None, :]) / sd[None, :]).astype(np.float32)\n"
CENTRED = "series = series.mask(series.rolling(96, center=True, min_periods=96).std() < 1e-6)"
TRAILING = "series = series.mask(series.rolling(96, center=False, min_periods=96).std() < 1e-6)"


def replace_once(text, old, new, label, path):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[FATAL] {os.path.basename(path)}: expected one '{label}', found {count}")
    return text.replace(old, new, 1)


for path in EVALUATORS:
    source = open(path, encoding="utf-8").read()
    source = replace_once(source, OLD_ANCHOR, NEW_ANCHOR, "anchor line", path)
    source = replace_once(source, Z_LINE, Z_LINE + ANCHOR_FUNCTION, "Z line", path)
    if CENTRED in source:
        source = replace_once(source, CENTRED, TRAILING, "centred flatline window", path)
    header = ('# Anchor-fix copy made by ' + os.path.abspath(__file__) + ' on 2026-09-24.\n'
              '# Original: ' + path + '\n')
    output = path[:-3] + "_anchorfix.py"
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(header + source)
    print("[anchorfix] wrote", output, flush=True)
