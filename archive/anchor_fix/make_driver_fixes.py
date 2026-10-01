"""Remove the one-off recovery wait from the copied replay driver so the anchor-fix rerun can start clean.

run_simulated_forecast.sh was written while LSTM seed 101 of the first attempt was already running, so it
waits for that archive before doing anything. In this fresh experiment nothing writes that file first, and the
driver would wait forever. The wait block is removed with a counted substitution. The LSTM seed 101 run then
happens in the driver's own LSTM loop, as for the other seeds.

run_simulated_forecast_remaining.sh, run_simulated_forecast_stgnn_after_driver.sh, summarize_simulated_forecast_when_ready.sh,
summarize_simulated_forecast_when_valid.sh and wait_for_jobs.sh are recovery scripts from the first attempt. The master
driver run_all.sh does not call them.
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(SCRIPTS, "run_simulated_forecast.sh")
WAIT_BLOCK = '''# Seed 101 of the LSTM may already be running from the first attempt. Wait for its archive
# instead of starting a second copy that would write the same file.
LSTM101="${RUN_DIR}/lstm_seed101_hrrr_jan_aug_postproc.npz"
if [ ! -f "${LSTM101}" ]
then
  echo "[driver] waiting for ${LSTM101}"
  while [ ! -f "${LSTM101}" ]
  do
    sleep 20
  done
  echo "[driver] found ${LSTM101}"
fi

'''

text = open(DRIVER, encoding="utf-8").read()
count = text.count(WAIT_BLOCK)
if count != 1:
    raise SystemExit(f"[FATAL] expected one wait block in {DRIVER}, found {count}")
text = text.replace(WAIT_BLOCK, "", 1)
with open(DRIVER, "w", encoding="utf-8") as handle:
    handle.write(text)
print("[driver-fix] removed the LSTM seed 101 wait block from", DRIVER)
