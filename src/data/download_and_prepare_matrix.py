import os
import io
import re
import csv
import glob
import time

# Hardcoded coordinates extracted from node_coords_2.csv
HARDCODED_NODE_COORDS = {
    "BCPD4501": (30.18982685, -90.78704437),
    "BCPD4502": (30.18982685, -90.78704437),
    "BCPD4503": (30.18982685, -90.78704437),
    "BCRA1299": (30.19857101, -90.92025409),
    "BCRA2400": (30.19720828, -90.88658803),
    "BCRA6146": (30.15483835, -90.70788572),
    "BCRU0003": (29.99327220, -90.25902750),
    "BCRU4350": (30.17077864, -90.91872060),
    "BCRU4420": (30.17327860, -90.86121900),
    "BCSA1299": (30.19857101, -90.92025409),
    "BCSA2400": (30.19720828, -90.88658803),
    "BCSA4409": (30.17488173, -90.87128322),
    "BCSA4419": (30.18912582, -90.78247457),
    "BCSA6146": (30.15483835, -90.70788572),
    "BCSU0001": (30.03013889, -90.03416667),
    "BCSU0002": (30.02666670, -90.11333330),
    "BCSU0005": (30.19963889, -90.12275000),
    "BCSU4350": (30.17077864, -90.91872060),
    "BCSU4419": (30.17327860, -90.86121900),
    "BCSU4420": (30.17327860, -90.86121900),
    "BCWU0001": (30.03013889, -90.03416667),
    "BCWU0002": (30.02666670, -90.11333330),
    "BCWU0003": (29.99327220, -90.25902750),
    "BMRA0347": (30.34230453, -90.99287383),
    "BMRA1292": (30.32375113, -91.01750926),
    "BMRA3548": (30.30582837, -90.95888699),
    "BMRU0004": (30.53291670, -91.14988890),
    "BMRU0005": (30.35519288, -91.01510220),
    "BMRU0006": (30.38685874, -91.00732440),
    "BMRU0008": (30.46407900, -90.99038000),
    "BMRU0009": (30.51268900, -91.07371580),
    "BMRU0581": (30.33686010, -90.96898970),
    "BMRU2210": (30.32138889, -91.02094440),
    "BMRU2620": (30.32055556, -90.95416670),
    "BMSA0347": (30.34230453, -90.99287383),
    "BMSA1291": (30.32314671, -91.01813352),
    "BMSA1292": (30.32375113, -91.01750926),
    "BMSA2200": (30.32319166, -91.01808229),
    "BMSA2201": (30.32319166, -91.01808229),
    "BMSA3548": (30.30582837, -90.95888699),
    "BMST1292": (30.32375113, -91.01750926),
    "BMST1293": (30.32375113, -91.01750926),
    "BMST2109": (30.32319166, -91.01808229),
    "BMSU0005": (30.35519288, -91.01510220),
    "BMSU0006": (30.38685874, -91.00732440),
    "BMSU0007": (30.44566667, -91.19155560),
    "BMSU0008": (30.46407900, -90.99038000),
    "BMSU0009": (30.51268900, -91.07371580),
    "BMSU0011": (30.53491060, -90.98065790),
    "BMSU0012": (30.35852615, -91.10816030),
    "BMSU0013": (30.38241445, -91.09427120),
    "BMSU0014": (30.41796914, -91.09149350),
    "BMSU0015": (30.40491390, -91.10343820),
    "BMSU0581": (30.33686010, -90.96898970),
    "BMSU0853": (30.34047129, -90.91732160),
    "BMSU2119": (30.32125000, -91.02072220),
    "BMSU2210": (30.32138889, -91.02094440),
    "BMSU2211": (30.32352700, -91.01815760),
    "BMSU2620": (30.32055556, -90.95416670),
    "BMWU0004": (30.53291670, -91.14988890),
    "HBPD1374": (30.31872988, -90.85979992),
    "HBPD1384": (30.31872988, -90.85979992),
    "HBRA1384": (30.31767584, -90.85969634),
    "HBRA2096": (30.30027901, -90.92165108),
    "HBRU0001": (30.33269390, -90.85204190),
    "HBRU0398": (30.33269390, -90.85204190),
    "HBRU3213": (30.29744160, -90.88371980),
    "HBRU4751": (30.27547330, -90.77926160),
    "HBSA1384": (30.31767584, -90.85969634),
    "HBSA2096": (30.30027901, -90.92165108),
    "HBSU0001": (30.33269390, -90.85204190),
    "HBSU0398": (30.33269390, -90.85204190),
    "HBSU1384": (30.31825000, -90.85986110),
    "HBSU1385": (30.31825000, -90.85986110),
    "HBSU3213": (30.29744160, -90.88371980),
    "HBSU4751": (30.27547330, -90.77926160),
    "HBSU6714": (30.30769444, -90.60872220),
    "MBPD9909": (30.18909282, -90.78793138),
    "MBPD9919": (30.18909282, -90.78793138),
    "MBPD9929": (30.18909282, -90.78793138),
    "MBPD9939": (30.18909282, -90.78793138),
    "MBPD9949": (30.18909282, -90.78793138),
    "MBPD9959": (30.18909282, -90.78793138),
    "MBPD9969": (30.18909282, -90.78793138),
    "MBRA3301": (30.27902075, -90.97138739),
    "MBRA4314": (30.26205752, -90.96329673),
    "MBRA4413": (30.26219002, -90.93989808),
    "MBRA4569": (30.25476101, -90.89630408),
    "MBRA5275": (30.23854102, -90.98929409),
    "MBRA5715": (30.24697543, -90.85215386),
    "MBRA5881": (30.23707875, -90.83515233),
    "MBRA6410": (30.23262230, -90.94691520),
    "MBRA6512": (30.23220133, -90.91392164),
    "MBRA6561": (30.22480049, -90.91722192),
    "MBRA7806": (30.21869726, -90.82111059),
    "MBRA8344": (30.19888900, -90.96388900),
    "MBRA8949": (30.18909312, -90.78793089),
    "MBRU3561": (30.26944444, -90.91694440),
    "MBRU4314": (30.26216480, -90.96344420),
    "MBRU5688": (30.23741844, -90.87009720),
    "MBRU6557": (30.22741860, -90.89954250),
    "MBRU8999": (30.18936433, -90.78620550),
    "MBSA3301": (30.27902075, -90.97138739),
    "MBSA4314": (30.26205752, -90.96329673),
    "MBSA4413": (30.26219002, -90.93989808),
    "MBSA4569": (30.25476101, -90.89630408),
    "MBSA5275": (30.23854102, -90.98929409),
    "MBSA5584": (30.23693831, -90.90877152),
    "MBSA5585": (30.23696397, -90.90866558),
    "MBSA5715": (30.24697543, -90.85215386),
    "MBSA5881": (30.23707875, -90.83515233),
    "MBSA6410": (30.23262230, -90.94691520),
    "MBSA6512": (30.23220133, -90.91392164),
    "MBSA6561": (30.22480049, -90.91722192),
    "MBSA7806": (30.21869726, -90.82111059),
    "MBSA8344": (30.19888900, -90.96388900),
    "MBSA9909": (30.18909312, -90.78793089),
    "MBSA9919": (30.18909312, -90.78793089),
    "MBST9908": (30.18909282, -90.78793138),
    "MBST9909": (30.18909282, -90.78793138),
    "MBST9918": (30.18909282, -90.78793138),
    "MBST9919": (30.18909282, -90.78793138),
    "MBST9928": (30.18909282, -90.78793138),
    "MBST9929": (30.18909282, -90.78793138),
    "MBST9938": (30.18909282, -90.78793138),
    "MBST9939": (30.18909282, -90.78793138),
    "MBST9948": (30.18909282, -90.78793138),
    "MBST9949": (30.18909282, -90.78793138),
    "MBSU0001": (30.30769444, -90.60872220),
    "MBSU0002": (30.28356667, -90.39783890),
    "MBSU0003": (30.22333333, -90.64000000),
    "MBSU3561": (30.26944444, -90.91694440),
    "MBSU4314": (30.26216480, -90.96344420),
    "MBSU5688": (30.23741844, -90.87009720),
    "MBSU6557": (30.22741860, -90.89954250),
    "MBSU8999": (30.18936433, -90.78620550),
    "MBSU9000": (30.18936433, -90.78620550),
    "MBSU9909": (30.18925000, -90.78127780),
    "MBWS5594": (30.23693831, -90.90877152),
    "MBWS6504": (30.23693831, -90.90877152),
    "BCRU0001": (30.03013889, -90.03416667),
}
print("[CFG] HARDCODED_NODE_COORDS count:", len(HARDCODED_NODE_COORDS))
import json
import sys
import warnings
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

print("[COMPLETED] LOGGING MODE             : line-buffered stdout/stderr")
print("[COMPLETED] SCRIPT_PATH              :", os.path.abspath(__file__))
print("[COMPLETED] CWD_AT_START             :", os.getcwd())

# =============================================================================
# CONFIG (single section, no hidden assumptions later)
# =============================================================================

# Base paths
BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, "Raw")
CLEAN_DIR = os.path.join(BASE_DIR, "Cleaned")
USGS_RAW_DIR = os.path.join(BASE_DIR, "USGS_Raw")
USGS_CLEANED_DIR = os.path.join(BASE_DIR, "USGS_Cleaned")
NOAA_RAW_DIR = RAW_DIR
MASTER_OUT_DIR = os.path.join(BASE_DIR, "Prepared_Global_Matrix")
POSTPROCESSED_DIR = os.path.join(BASE_DIR, "Postprocessed")
POSTPROCESSED_PLOT_DIR = os.path.join(BASE_DIR, "Postprocessed_Plots")
TRIMMED_DIR = os.path.join(BASE_DIR, "Postprocessed_Allstations")
POSTPROCESSED_SELECTED_DIR = os.path.join(BASE_DIR, "Postprocessed_Selectedstations")

# Timezone and resample rule
LOCAL_TZ = "America/Chicago"
RESAMPLE_RULE = "15min"
RESAMPLE_MINUTES = int(pd.Timedelta(RESAMPLE_RULE).total_seconds() / 60)

# Global run mode and universal dates
START_DATE_UTC = "2023-01-01T00:00:00Z"
END_DATE_UTC = os.environ.get("ALPHA_DOWNLOAD_END_UTC", "").strip() or None
CUTOFF_DATE_UTC = None
CUTOFF_RELAX_HOURS = 1
STIFF_START_DATE = pd.Timestamp("2025-10-21 15:30:00", tz="UTC") # this is the youngest data start
# FULL-HISTORY FIX (2026-06-09) + HARDCODE (2026-06-25): the all-stations matrix was once hard-floored to
# STIFF_START_DATE (2025-10-21), discarding ~2.8 yr of data even though raw CSVs go back to 2022-2023. The floor
# is now HARDCODED to 2023-01-01 (NO environment variable) — this runs on a public, shared HPC where an env-miss
# would silently fall back to the dangerous 2025-10-21 floor. Late-starting stations are flat-backfilled pre-start
# and the GNN trainer's isfinite+flatline screen masks those cells out of the loss. See
# INVESTIGATE_DQP80_DATA_TRIM_20260609.md. To change the floor, edit THIS line (do not reintroduce os.environ).
ALLSTN_GRID_FLOOR = pd.Timestamp("2023-01-01", tz="UTC")
print(f"[CFG] ALLSTN_GRID_FLOOR = {ALLSTN_GRID_FLOOR} (HARDCODED, no env)")
print(f"[CFG] ALPHA_DOWNLOAD_END_UTC = {END_DATE_UTC or 'current UTC time'}")
SELECTED_STATIONS_CUTOFF_DATE_UTC = "2023-01-01T00:00:00Z"
# HTTP behavior
USER_AGENT = "Farid-Preprocessing/1.0"
REQUEST_TIMEOUT = 60
RETRY_MAX = 5
RETRY_BACKOFF_SEC = 2
RETRY_STATUSES = [429, 500, 502, 503, 504]

# APG SCADA download config
APG_BASE_URL = "https://gis.ascensionparishla.gov/scada/data/"
APG_SLEEP_BETWEEN = 0.05
APG_INDEX_FILES = [
    ("sites_apg_rain.csv", "rain"),
    ("sites_apg_stream.csv", "stream"),
    ("sites_usgs_rain.csv", "rain"),
    ("sites_usgs_stream.csv", "stream"),
    ("sites_apg_pumps.csv", "pump"),
    ("sites_apg_gates.csv", "gate"),
    ("sites_apg_weirs.csv", "weir"),
]
APG_SUFFIX = {
    "rain": "_rain.csv",
    "stream": "_stream.csv",
    "pump": "_pump.csv",
    "gate": "_gate.csv",
    "weir": "_weir.csv",
}
# APG station-code pattern used to harvest codes out of the live catalogue files.
# TIGHTENED 2026-08-05 (FIX F4). The old pattern was r"\b[A-Z]{3,6}[A-Z0-9]{0,2}\d{3,4}\b", which is
# not a specification of anything — it just happened to match the codes. Verified against every live
# catalogue on 2026-08-05: old pattern 114 codes, new pattern 114 codes, NOTHING dropped and NOTHING
# gained, so this is behaviour-preserving today. It matters for the coming APG ID rename: applied to
# the new-format catalogues the loose pattern also matches "HWY621"/"HWY431" inside
# MB-STRM-U-BLAC-HWY621, which would be requested as HWY621_stream.csv and 404.
APG_CODE_RE = re.compile(r"\b(?:BC|BM|MB|HB)(?:SA|SU|RA|RU|WU|ST|PD|WS)\d{4}\b")
# The loose pattern is kept ONLY to report what the tightening rejects, so a real code can never
# disappear silently just because it does not fit the assumed shape.
APG_CODE_RE_LOOSE = re.compile(r"\b[A-Z]{3,6}[A-Z0-9]{0,2}\d{3,4}\b")
# SAFETY FLOOR (FIX F4). The scrape reads a file on APG's web server that we do not control. If APG
# republishes the catalogues without the legacy_name column, the harvest returns NOTHING, every APG
# station silently vanishes from the training matrix, the failed downloads only print
# "[UNSUCCESSFUL] GET failed" and continue, and the job still exits 0 and uploads a model trained
# without 29 stream / 21 rain / 13 gate / 12 pump / 4 weir feeds. That is the worst realistic failure
# in the whole ID migration, and it is completely silent. Counts below are what each catalogue
# actually yielded on 2026-08-05; the tolerance allows for a station being legitimately retired
# (e.g. MBSA4570, which APG told us to disregard) without allowing a collapse.
APG_INDEX_EXPECTED_CODES = {
    "sites_apg_rain.csv": 21,
    "sites_apg_stream.csv": 29,
    "sites_usgs_rain.csv": 13,
    "sites_usgs_stream.csv": 22,
    "sites_apg_pumps.csv": 12,
    "sites_apg_gates.csv": 13,
    "sites_apg_weirs.csv": 4,
}
APG_INDEX_FLOOR_TOLERANCE = 3
APG_TOTAL_MIN_TARGETS = 100          # today: 114

# ---- APG ID SCHEME (migration 2026-08-06) --------------------------------------------------------
# APG moved every feed onto a new identifier grammar -- BASIN-TYPE-OWNER-LOCATION-SUBLOCATION, e.g.
# BC-STRM-A-BOYL-BURNSI -- and onto new directories:
#
#     legacy : catalogues  https://gis.ascensionparishla.gov/scada/data/sites_*.csv
#              data        https://gis.ascensionparishla.gov/scada/data/<CODE><_kind>.csv
#     new    : catalogues  https://gis.ascensionparishla.gov/scada/data/share/input/sites_*.csv
#              data        https://gis.ascensionparishla.gov/scada/data/share/public/<NEWID>.csv
#
# ONLY THE URL CHANGES. The harvested legacy code stays the station's identity everywhere downstream
# -- the raw filename (RAW_DIR/<code>_<kind>.csv), the cleaned filename, and ultimately the training
# matrix column name (<code>_<vcol>). Renaming it would rename every column of the training matrix
# and invalidate the graph, gnn_meta.json, stage_to_rain_gauge.csv, the cap table and every stored weight.
#
# Rollback is one word: set APG_ID_SCHEME to "legacy", or export APG_ID_SCHEME=legacy.
APG_ID_SCHEME = os.environ.get("APG_ID_SCHEME", "new").strip().lower()
if APG_ID_SCHEME not in ("legacy", "new"):
    print("[WARNING] [APG-ID] Unknown APG_ID_SCHEME:", APG_ID_SCHEME, "-> falling back to 'legacy'")
    APG_ID_SCHEME = "legacy"
APG_NEW_DATA_BASE_URL = "https://gis.ascensionparishla.gov/scada/data/share/public/"
APG_NEW_INDEX_BASE_URL = "https://gis.ascensionparishla.gov/scada/data/share/input/"
APG_CROSSWALK_CSV = os.environ.get(
    "APG_CROSSWALK_CSV",
    "project/hpc/Training/apg_station_crosswalk.csv",
)

# APG's catalogues carry a `legacy_name` column, which is what the harvest above actually reads, so
# the same regex yields the same 114 legacy codes from either location -- verified 2026-08-06
# (catalogue_harvest_simulation.txt) with EXACTLY ONE exception:
#
#     APG lists the Sorrento rain series under legacy_name BCRU4419; our column is BCRU4420.
#
# It is one physical rain series on USGS 073802225. That site returns two stage series (0 =
# downstream = our BCSU4420, 1 = upstream = our BCSU4419) but only one rain series, and Dean asked us
# to map it to the UPSTREAM site, so APG named it BC-RAIN-U-BELL-SOPSUP to match BC-STRM-U-BELL-SOPSUP.
#
# Left unmapped this is NOT cosmetic: the harvest would emit BCRU4419, the trainer would write
# BCRU4419_rain.csv and produce a training-matrix column BCRU4419_rain_in that the model has never
# seen, while BCRU4420_rain_in -- the assigned rain gauge for BCSA4409, BCSU4419 and BCSU4420 in
# stage_to_rain_gauge.csv -- would disappear. Normalising APG's legacy_name back to our code keeps the
# matrix byte-identical.
APG_LEGACY_NAME_TO_OUR_CODE = {
    "BCRU4419": "BCRU4420",
}

APG_NEW_ID_BY_CODE = {}
if APG_ID_SCHEME == "new":
    try:
        _cw_df = pd.read_csv(APG_CROSSWALK_CSV, dtype=str).fillna("")
        for _, _cw_row in _cw_df.iterrows():
            _cw_code = str(_cw_row["station_code"]).strip()
            _cw_nid = str(_cw_row["new_id"]).strip()
            if _cw_nid != "" and str(_cw_row["id_source"]).strip() == "APG":
                APG_NEW_ID_BY_CODE[_cw_code] = _cw_nid
        print("[COMPLETED] [APG-ID] scheme='new' | crosswalk:", APG_CROSSWALK_CSV)
        print("[COMPLETED] [APG-ID] APG-hosted identifiers loaded:", len(APG_NEW_ID_BY_CODE))
    except Exception as _cw_exc:
        raise RuntimeError(
            f"[APG-ID][FATAL] APG_ID_SCHEME='new' but the crosswalk could not be read: "
            f"{APG_CROSSWALK_CSV} -- {_cw_exc!r}. Refusing to build a training matrix without a "
            f"complete station map. Set APG_ID_SCHEME=legacy to roll back."
        )
else:
    print("[COMPLETED] [APG-ID] scheme='legacy' | using the pre-migration URLs")


def apg_index_url_for(index_filename):
    """URL of one catalogue file, for the active ID scheme."""
    if APG_ID_SCHEME == "legacy":
        return APG_BASE_URL + index_filename
    return APG_NEW_INDEX_BASE_URL + index_filename


def apg_data_url_for(station_code, kind):
    """URL of one station's data CSV, for the active ID scheme. None if APG does not publish it."""
    if APG_ID_SCHEME == "legacy":
        return APG_BASE_URL + station_code + APG_SUFFIX[kind]
    new_id = APG_NEW_ID_BY_CODE.get(station_code)
    if not new_id:
        return None
    return APG_NEW_DATA_BASE_URL + new_id + ".csv"


# ---- GATE STATUS DECODE (work item W1, migration 2026-08-06) --------------------------------------
# APG's engineer moved the gates off the `Opened` tag onto a `Status` tag computed from BOTH the
# Opened and Closed limit switches (Dean, 2026-08-05):
#
#     3 = Error   both limit switches asserted -> the reading is meaningless
#     2 = Closed
#     1 = Opened
#     0 = Motion / Transition
#
# Without the decode, the existing `invalid = ~s.isin([0.0, 1.0])` below sends BOTH 2 (Closed) and
# 3 (Error) to NaN, and the ffill that follows then carries the PREVIOUS state forward -- so a gate
# that closes keeps training the model as OPEN until the next Opened event arrives. The serving side
# is worse still: its round()/clip(0,1) maps 2 and 3 onto 1.
#
# Verified against the full legacy/new overlap (gate_status_dictionary_check.txt): 13 gates,
# 711,762 shared 15-minute steps, 99.58% agreement with the legacy tag, ZERO inversions. The three
# Bayou Manchac gates still emit only 0/1 and the same rule leaves them unchanged.
#
# This must stay byte-identical in meaning to the block in Inference/Download_Operational.py --
# a difference between the two IS a train/serve skew.
GATE_STATUS_DECODE_ENABLE = os.environ.get("GATE_STATUS_DECODE", "on").strip().lower() in (
    "on", "1", "true", "yes",
)
GATE_STATUS_DECODE = {0.0: 0.0, 1.0: 1.0, 2.0: 0.0, 3.0: np.nan}
STRUCT_QC_ENABLE = os.environ.get("STRUCT_OBS_QC", "on").strip().lower() in (
    "on", "1", "true", "yes",
)
GATE_VALID_STATUS_VALUES = (0.0, 1.0, 2.0, 3.0)
PUMP_VALID_STATUS_VALUES = (0.0, 1.0)

# USGS IV download config
USGS_IV_ENDPOINT = "https://waterservices.usgs.gov/nwis/iv/"
USGS_CHUNK_DAYS = 60
USGS_PAUSE_SEC = 0.30
USGS_PARAM_FOR_KIND = {
    "rain": "00045",
    "stream": "00065",
    "wind": "00035",
}
USGS_MAP = [
    {"apg_filename": "BMRU2210_rain.csv", "kind": "rain", "site": "07378746"},
    {"apg_filename": "BMRU0581_rain.csv", "kind": "rain", "site": "07380102"},
    {"apg_filename": "BMRU2620_rain.csv", "kind": "rain", "site": "07380106"},
    {"apg_filename": "HBRU3213_rain.csv", "kind": "rain", "site": "07380126"},
    {"apg_filename": "HBRU0398_rain.csv", "kind": "rain", "site": "07380120"},
    {"apg_filename": "HBRU4751_rain.csv", "kind": "rain", "site": "07380200"},
    {"apg_filename": "MBRU4314_rain.csv", "kind": "rain", "site": "0738022295"},
    {"apg_filename": "MBRU3561_rain.csv", "kind": "rain", "site": "0738022395"},
    {"apg_filename": "MBRU5688_rain.csv", "kind": "rain", "site": "073802245"},
    {"apg_filename": "MBRU6557_rain.csv", "kind": "rain", "site": "073802273"},
    {"apg_filename": "MBRU8999_rain.csv", "kind": "rain", "site": "073802282"},
    {"apg_filename": "BCRU4350_rain.csv", "kind": "rain", "site": "073802220"},
    {"apg_filename": "BCRU4420_rain.csv", "kind": "rain", "site": "073802225"},
    {"apg_filename": "BMRU0005_rain.csv", "kind": "rain", "site": "07379075"},
    {"apg_filename": "BMRU0006_rain.csv", "kind": "rain", "site": "07378722"},
    {"apg_filename": "BCWU0001_wind.csv", "kind": "wind", "site": "073802332"},
    # NEW RAIN STATIONS
    {"apg_filename": "HBRU0001_rain.csv", "kind": "rain", "site": "07380120"},
    #{"apg_filename": "BMRU0008_rain.csv", "kind": "rain", "site": "07378500"},
    #{"apg_filename": "BMRU0009_rain.csv", "kind": "rain", "site": "07378000"},
    {"apg_filename": "BMSU2119_stream.csv", "kind": "stream", "site": "07378745", "idx": 0},
    {"apg_filename": "BMSU2211_stream.csv", "kind": "stream", "site": "07378748", "idx": 0},
    {"apg_filename": "BMSU2210_stream.csv", "kind": "stream", "site": "07378746", "idx": 0},
    {"apg_filename": "BMSU0581_stream.csv", "kind": "stream", "site": "07380102", "idx": 0},
    {"apg_filename": "BMSU0853_stream.csv", "kind": "stream", "site": "07380101", "idx": 0},
    {"apg_filename": "BMSU2620_stream.csv", "kind": "stream", "site": "07380106", "idx": 0},
    {"apg_filename": "HBSU3213_stream.csv", "kind": "stream", "site": "07380126", "idx": 0},
    {"apg_filename": "HBSU1384_stream.csv", "kind": "stream", "site": "07380127", "idx": 0},
    {"apg_filename": "HBSU1385_stream.csv", "kind": "stream", "site": "07380127", "idx": 1},
    {"apg_filename": "HBSU0398_stream.csv", "kind": "stream", "site": "07380120", "idx": 0},
    {"apg_filename": "HBSU4751_stream.csv", "kind": "stream", "site": "07380200", "idx": 0},
    {"apg_filename": "HBSU6714_stream.csv", "kind": "stream", "site": "07380215", "idx": 0},
    {"apg_filename": "MBSU4314_stream.csv", "kind": "stream", "site": "0738022295", "idx": 0},
    {"apg_filename": "MBSU3561_stream.csv", "kind": "stream", "site": "0738022395", "idx": 0},
    {"apg_filename": "MBSU5688_stream.csv", "kind": "stream", "site": "073802245", "idx": 0},
    {"apg_filename": "MBSU6557_stream.csv", "kind": "stream", "site": "073802273", "idx": 0},
    {"apg_filename": "MBSU8999_stream.csv", "kind": "stream", "site": "073802282", "idx": 1},
    {"apg_filename": "MBSU9000_stream.csv", "kind": "stream", "site": "073802282", "idx": 0},
    {"apg_filename": "MBSU9909_stream.csv", "kind": "stream", "site": "073802284", "idx": 0},
    {"apg_filename": "BCSU4350_stream.csv", "kind": "stream", "site": "073802220", "idx": 0},
    {"apg_filename": "BCSU4419_stream.csv", "kind": "stream", "site": "073802225", "idx": 1},
    {"apg_filename": "BCSU4420_stream.csv", "kind": "stream", "site": "073802225", "idx": 0},
    {"apg_filename": "BCSU0001_stream.csv", "kind": "stream", "site": "073802332", "idx": 0},
    {"apg_filename": "BCSU0005_stream.csv", "kind": "stream", "site": "301200090072400", "idx": 0},
    {"apg_filename": "MBSU0001_stream.csv", "kind": "stream", "site": "07380215", "idx": 0},
    {"apg_filename": "MBSU0002_stream.csv", "kind": "stream", "site": "073802302", "idx": 0},
    {"apg_filename": "MBSU0003_stream.csv", "kind": "stream", "site": "301324090382400", "idx": 0},
    {"apg_filename": "BMSU0005_stream.csv", "kind": "stream", "site": "07379075", "idx": 0},
    {"apg_filename": "BMSU0006_stream.csv", "kind": "stream", "site": "07378722", "idx": 0},
    # NEW STREAM STATIONS
    #{"apg_filename": "BMSU0007_stream.csv", "kind": "stream", "site": "07374000", "idx": 0},
    {"apg_filename": "HBSU0001_stream.csv", "kind": "stream", "site": "07380120", "idx": 0},
    #{"apg_filename": "BMSU0008_stream.csv", "kind": "stream", "site": "07378500", "idx": 0},
    #{"apg_filename": "BMSU0009_stream.csv", "kind": "stream", "site": "07378000", "idx": 0},
    #{"apg_filename": "BMSU0011_stream.csv", "kind": "stream", "site": "07377300", "idx": 0},
    {"apg_filename": "BMSU0012_stream.csv", "kind": "stream", "site": "07378810", "idx": 0},
    {"apg_filename": "BMSU0013_stream.csv", "kind": "stream", "site": "07379960", "idx": 0},
    {"apg_filename": "BMSU0014_stream.csv", "kind": "stream", "site": "07379100", "idx": 0},
    {"apg_filename": "BMSU0015_stream.csv", "kind": "stream", "site": "07379050", "idx": 0},

]

# USGS datum (NAVD88) config
# Source CSV used to manually extract hardcoded rows below.
USGS_OFFSETS_SOURCE_CSV_LOCAL = os.path.join(BASE_DIR, "usgs_navd88_offsets.csv")
USGS_OFFSETS_SOURCE_CSV_FALLBACK = "project/hpc_home/Ascension_AI_Project/best_preprocessing/Pickle_Without_Trimming/USGS_Raw/usgs_navd88_offsets.csv"
if os.path.exists(USGS_OFFSETS_SOURCE_CSV_LOCAL):
    USGS_OFFSETS_SOURCE_CSV = USGS_OFFSETS_SOURCE_CSV_LOCAL
else:
    USGS_OFFSETS_SOURCE_CSV = USGS_OFFSETS_SOURCE_CSV_FALLBACK
print("[CONFIG] USGS_OFFSETS_SOURCE_CSV:", USGS_OFFSETS_SOURCE_CSV)
# Hardcoded directly from USGS_OFFSETS_SOURCE_CSV (no runtime CSV read/download/build).
USGS_OFFSETS_HARDCODED_ROWS = [
    {"file_stem": "BMSU2119", "site_no": "07378745", "index": 0, "offset_ft": "-0.500", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU2211", "site_no": "07378748", "index": 0, "offset_ft": "-0.500", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU2210", "site_no": "07378746", "index": 0, "offset_ft": "-0.500", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0581", "site_no": "07380102", "index": 0, "offset_ft": "0.710", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0853", "site_no": "07380101", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BMSU2620", "site_no": "07380106", "index": 0, "offset_ft": "4.840", "source": "overlap_iv", "note": ""},
    {"file_stem": "HBSU3213", "site_no": "07380126", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "HBSU1384", "site_no": "07380127", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "HBSU1385", "site_no": "07380127", "index": 1, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "HBSU0398", "site_no": "07380120", "index": 0, "offset_ft": "-1.360", "source": "overlap_iv", "note": ""},
    {"file_stem": "HBSU4751", "site_no": "07380200", "index": 0, "offset_ft": "0", "source": "overlap_iv", "note": ""},
    {"file_stem": "HBSU6714", "site_no": "07380215", "index": 0, "offset_ft": "-1.380", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU4314", "site_no": "0738022295", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU3561", "site_no": "0738022395", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU5688", "site_no": "073802245", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU6557", "site_no": "073802273", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU8999", "site_no": "073802282", "index": 1, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "MBSU9000", "site_no": "073802282", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "MBSU9909", "site_no": "073802284", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BCSU4350", "site_no": "073802220", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "BCSU4419", "site_no": "073802225", "index": 1, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BCSU4420", "site_no": "073802225", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BCSU0001", "site_no": "073802332", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BCSU0005", "site_no": "301200090072400", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "MBSU0001", "site_no": "07380215", "index": 0, "offset_ft": "-1.380", "source": "overlap_iv", "note": ""},
    {"file_stem": "MBSU0002", "site_no": "073802302", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "MBSU0003", "site_no": "301324090382400", "index": 0, "offset_ft": "-0.650", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0005", "site_no": "07379075", "index": 0, "offset_ft": "", "source": "inventory_fetch_failed", "note": "no_offset_found_or_other_datum"},
    {"file_stem": "BMSU0006", "site_no": "07378722", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "HBSU0001", "site_no": "07380120", "index": 0, "offset_ft": "-1.360", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0012", "site_no": "07378810", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0013", "site_no": "07379960", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0014", "site_no": "07379100", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
    {"file_stem": "BMSU0015", "site_no": "07379050", "index": 0, "offset_ft": "0.000", "source": "overlap_iv", "note": ""},
]
USGS_DATUM_OFFSET_ABS_MAX = 100.0

# NOAA ASOS download config
ASOS_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ASOS_STATION_IDS = ["BTR", "MSY", "NEW", "REG", "NEW_LAKEFRONT"]
# Keys are LOCAL aliases; "asos_id" (optional, defaults to the key) is the real IEM/ASOS station.
# "rain_code": None -> wind only (do NOT invent new rain columns; the rain station set is part of
# the model's feature contract and changing it would require a retrain + revalidation).
ASOS_STATION_MAP = {
    "MSY": {"wind_code": "BCWU0003", "rain_code": "BCRU0003"},
    "BTR": {"wind_code": "BMWU0004", "rain_code": "BMRU0004"},
    "NEW": {"wind_code": "BCWU0001", "rain_code": "BCRU0001"},
    # UNION-5 wind channel (deployed 2026-07-23): KREG = the only wind sensor INSIDE Ascension
    # Parish; NEW Lakefront = on Lake Pontchartrain, measuring the surge-setup fetch. Both are
    # wind-only additions. NEW appears twice on purpose: it still backfills the legacy BCWU0001
    # column, and now also populates its own BCWU0006 column used by the wind channel.
    "REG": {"asos_id": "REG", "wind_code": "BCWU0005", "rain_code": None},
    "NEW_LAKEFRONT": {"asos_id": "NEW", "wind_code": "BCWU0006", "rain_code": None},
}

# NOAA COOPS config
COOPS_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_STATION_ID = "8761927"
COOPS_WIND_CODE = "BCWU0002"
COOPS_STREAM_CODE = "BCSU0002"
COOPS_NAVD_OFFSET_FT = 0.23
COOPS_MAX_DAYS = 30

# RAW sanity checks (very loose, mainly for warnings)
RAIN_MAX_PER_INTERVAL = 10.0
STAGE_MIN_FT = -10.0
STAGE_MAX_FT = 30.0
WEIR_MIN_FT = -10.0
WEIR_MAX_FT = 30.0
WIND_MAX_MPH = 200.0

# Clean mapping
CLEAN_INPUT_DIRS = [RAW_DIR, USGS_RAW_DIR, USGS_CLEANED_DIR]
CLEAN_COLMAP = {
    "rain": "rain_in",
    "stream": "stage_ft",
    "pump": "status",
    "gate": "status",
    "weir": "elev_ft",
    "wind": "wind_mph",
}

# CLEAN sanity (loose, but enforce basic correctness)
CLEAN_SANITY = {
    "rain_in_min_per_15m": 0.0,
    # QC CLIP (deployed 2026-07-07, from Experiments/FIX_REPLAY_20260706): 999.0 -> 3.0. The old
    # value was effectively OFF — BMRU0005 reported 106.1 in in ONE 15-min step on 2026-07-05 and
    # 999.0 would have let it straight into the training matrix. History check: only 21 of
    # 2,797,348 gauge-steps (2023-01 -> 2026-07) exceed 3.0 in, all sensor faults or catch-up dumps.
    "rain_in_max_per_15m": 3.0,
    "stage_ft_min": -999.0,
    "stage_ft_max": 999.0,
    "weir_ft_min": -999.0,
    "weir_ft_max": 999.0,
    "wind_mph_min": 0.0,
    "wind_mph_max": 999.0,
}

FEATURE_KINDS = ["stream", "rain", "wind","pump", "gate", "weir"]

# =============================================================================
# QARTOD PRIMARY FLAGS (IOOS):
# 1 = PASS (good)
# 3 = SUSPECT / of high interest
# 4 = FAIL (bad)
# 9 = MISSING
# See IOOS QARTOD flags manual for definitions.
# =============================================================================

# What gets nuked (set to NaN) before interpolation
NUKE_QARTOD_FLAGS = [4]

# Global stage sentinel handling (optional)
GLOBAL_STAGE_SENTINELS = [-9999.0, 9999.0, -6.0]  # keep -6 here if you want global nuke behavior
GLOBAL_STAGE_SENTINEL_MIN_RUN = 8                 # only nuke long runs (except station override below)

# Station-specific overrides (THIS is your "BCSA1299 nuke -6 no matter what")
STATION_QC_OVERRIDES = {
    "BCSA1299": {
        "stage_sentinels": [-6.0],
        "sentinel_min_run": 1,     # nuke even single sample
        "gross_min_ft": -6.0,      # optional: treat anything below as fail too
        "gross_max_ft": 25.0,      # optional
    }
}

# Stage QARTOD-ish thresholds (keep simple)
STAGE_GROSS_MIN_FT = -6.0
STAGE_GROSS_MAX_FT = 25.0

# Flatline: suspect then fail
STAGE_FLAT_EPS = 1e-4
STAGE_FLAT_SUSPECT_MIN_RUN = 96    # 24h at 15-min
STAGE_FLAT_FAIL_MIN_RUN = 192      # 48h at 15-min

# Rate of change: suspect then fail (ft per sample)
STAGE_ROC_SUSPECT_FT = 2.0
STAGE_ROC_FAIL_FT = 4.0

# Interp limits (samples)
POST_INTERP_LIMIT = 16             # 4 hours for 15-min
POST_PAD_SAMPLES = 4               # pad around FAIL blocks

# Rain postprocess
RAIN_GROSS_MIN_IN = 0.0
RAIN_GROSS_MAX_IN = 10.0
RAIN_INTERP_LIMIT = 16

# Plot switches
POSTPROCESS_PLOTS = True

# =============================================================================
# Output folders
# =============================================================================

RUN_TS_UTC = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_STAMP = RUN_TS_UTC

RUN_FEATURE_DIR = MASTER_OUT_DIR
RUN_EXOG_DIR = os.path.join(RUN_FEATURE_DIR, "exog")
RUN_QC_DIR = os.path.join(RUN_FEATURE_DIR, "qc")

for d in [
    RAW_DIR,
    CLEAN_DIR,
    USGS_RAW_DIR,
    USGS_CLEANED_DIR,
    POSTPROCESSED_DIR,
    POSTPROCESSED_PLOT_DIR,
    TRIMMED_DIR,
    POSTPROCESSED_SELECTED_DIR,
    RUN_FEATURE_DIR,
    RUN_EXOG_DIR,
    RUN_QC_DIR,
]:
    os.makedirs(d, exist_ok=True)

print("[COMPLETED] CONFIG START")
print("[COMPLETED] BASE_DIR                 :", BASE_DIR)
print("[COMPLETED] RUN_STAMP                :", RUN_STAMP)
print("[COMPLETED] RAW_DIR                  :", RAW_DIR)
print("[COMPLETED] CLEAN_DIR                :", CLEAN_DIR)
print("[COMPLETED] USGS_RAW_DIR             :", USGS_RAW_DIR)
print("[COMPLETED] USGS_CLEANED_DIR         :", USGS_CLEANED_DIR)
print("[COMPLETED] POSTPROCESSED_DIR        :", POSTPROCESSED_DIR)
print("[COMPLETED] POSTPROCESSED_PLOT_DIR   :", POSTPROCESSED_PLOT_DIR)
print("[COMPLETED] TRIMMED_DIR              :", TRIMMED_DIR)
print("[COMPLETED] POSTPROCESSED_SELECTED_DIR:", POSTPROCESSED_SELECTED_DIR)
print("[COMPLETED] MASTER_OUT_DIR           :", MASTER_OUT_DIR)
print("[COMPLETED] LOCAL_TZ                 :", LOCAL_TZ)
print("[COMPLETED] RESAMPLE_RULE            :", RESAMPLE_RULE)
print("[COMPLETED] START_DATE_UTC           :", START_DATE_UTC)
print("[COMPLETED] END_DATE_UTC             :", END_DATE_UTC)
print("[COMPLETED] CUTOFF_DATE_UTC          :", CUTOFF_DATE_UTC)
print("[COMPLETED] CONFIG END")

# =============================================================================
# Helpers (big ones only)
# =============================================================================

def parse_utc(value, default_now):
    if value is None:
        if default_now:
            return pd.Timestamp(datetime.now(timezone.utc))
        return None

    s = str(value).strip()
    if s == "":
        if default_now:
            return pd.Timestamp(datetime.now(timezone.utc))
        return None

    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")

    ts = pd.Timestamp(s)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def request_with_retry(session, method, url, params=None, stream=False, timeout=60):
    attempt = 0
    while attempt <= RETRY_MAX:
        attempt = attempt + 1
        try:
            r = session.request(method, url, params=params, stream=stream, timeout=timeout)
            if r.status_code in RETRY_STATUSES:
                print("[WARNING] HTTP", method, url, "status", r.status_code, "retry", attempt)
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue
            return r
        except Exception as exc:
            print("[WARNING] HTTP", method, url, "exception", repr(exc), "retry", attempt)
            time.sleep(RETRY_BACKOFF_SEC * attempt)
    return None


def run_length(mask):
    rid = (mask != mask.shift(1, fill_value=False)).cumsum()
    return mask.groupby(rid).transform("size")


def detect_kind_from_filename(path):
    name = os.path.basename(path).lower()
    if name.endswith("_rain.csv"):
        return "rain"
    if name.endswith("_stream.csv"):
        return "stream"
    if name.endswith("_pump.csv"):
        return "pump"
    if name.endswith("_gate.csv"):
        return "gate"
    if name.endswith("_weir.csv"):
        return "weir"
    if name.endswith("_wind.csv"):
        return "wind"
    return None


def validate_raw_frame(df, key, kind):
    ok = True
    if "timestamp_utc" not in df.columns:
        print("[ERROR] Missing timestamp_utc for", key)
        ok = False
    if "value_raw" not in df.columns:
        print("[ERROR] Missing value_raw for", key)
        ok = False
    if not ok:
        return False

    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts > 0:
        print("[WARNING] Bad timestamps for", key, "count", bad_ts)

    vals = pd.to_numeric(df["value_raw"], errors="coerce")
    bad_vals = int(vals.isna().sum())
    if bad_vals > 0:
        print("[WARNING] Non-numeric values for", key, "count", bad_vals)

    if kind == "rain":
        neg = int((vals < 0).sum())
        high = int((vals > RAIN_MAX_PER_INTERVAL * 10).sum())
        if neg > 0 or high > 0:
            print("[WARNING] Rain out-of-range for", key, "neg", neg, "high", high)

    if kind in ["stream", "weir"]:
        low = int((vals < STAGE_MIN_FT).sum())
        high = int((vals > STAGE_MAX_FT).sum())
        if low > 0 or high > 0:
            print("[WARNING] Stage out-of-range for", key, "low", low, "high", high)

    if kind == "wind":
        neg = int((vals < 0).sum())
        high = int((vals > WIND_MAX_MPH * 2).sum())
        if neg > 0 or high > 0:
            print("[WARNING] Wind out-of-range for", key, "neg", neg, "high", high)

    if kind in ["pump", "gate"]:
        invalid = int((~vals.isin([0, 1]) & vals.notna()).sum())
        if invalid > 0:
            print("[WARNING] Binary state invalid for", key, "count", invalid)

    return True


def parse_timestamp_column_to_utc(df, ts_col):
    ts_raw = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
    idx = pd.DatetimeIndex(ts_raw)

    tz_present = pd.api.types.is_datetime64tz_dtype(idx)
    if not tz_present:
        try:
            idx_local = idx.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
        except Exception as exc:
            print("[WARNING] tz_localize infer failed:", repr(exc))
            idx_local = idx.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
            mask_nat = idx_local.isna()
            if mask_nat.any():
                df = df.loc[~mask_nat].copy()
                idx_local = idx_local[~mask_nat]
        idx_utc = idx_local.tz_convert("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    return df, idx_utc


def keep_existing_record(existing_df, code, kind):
    """On 'no new data', re-emit an already-saved [timestamp_utc, value_raw] series into raw_records."""
    if existing_df is None or "timestamp_utc" not in existing_df.columns or "value_raw" not in existing_df.columns:
        return
    df_keep = existing_df[["timestamp_utc", "value_raw"]].copy()
    df_keep["timestamp_utc"] = pd.to_datetime(df_keep["timestamp_utc"], utc=True, errors="coerce")
    df_keep["value_raw"] = pd.to_numeric(df_keep["value_raw"], errors="coerce")
    df_keep = df_keep.dropna(subset=["timestamp_utc", "value_raw"])
    df_keep = df_keep.sort_values("timestamp_utc")
    # DEAD-GAUGE ALARM (FIX F11, 2026-08-05) — OBSERVABILITY ONLY, behaviour deliberately unchanged.
    # Re-emitting the last saved series is right for a brief telemetry outage and wrong for a RETIRED
    # station: MBRA8949 last reported 2026-05-06, its endpoint now returns HTTP 404, and this function
    # quietly re-injected its frozen May values into the training matrix every night for three months.
    # Nothing noticed, because a dead gauge and a quiet gauge look identical here. It is still the
    # assigned nearest rain gauge for graph nodes MBSA9909/MBSA9919 (stage_to_rain_gauge.csv), so those two
    # in-parish Marvin Braud nodes have carried zero observed rain since May.
    # We do NOT drop the column: stage_to_rain_gauge.csv is a MODEL asset, so repointing it needs a retrain
    # plus revalidation, and APG has been asked when/why the gauge was retired. Until that comes back,
    # keep the data flowing exactly as before and make the situation impossible to miss in the log.
    if len(df_keep) > 0:
        last_ts = pd.Timestamp(df_keep["timestamp_utc"].max())
        age_days = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 86400.0
        if age_days > 30.0:
            print(f"[WARNING] [DEAD-GAUGE] {code} ({kind}): re-emitting a series whose newest reading "
                  f"is {age_days:.1f} DAYS old (last {last_ts}). The station may be RETIRED, not just "
                  f"quiet. Behaviour unchanged by design -- see Finding F11.")
        elif age_days > 7.0:
            print(f"[WARNING] [STALE] {code} ({kind}): re-emitting a series {age_days:.1f} days old "
                  f"(last {last_ts}).")
    raw_records.append({"station": code, "kind": kind, "df": df_keep.copy()})


def qartod_primary_stage(series, station_code):
    """
    Returns:
      flag_primary: pd.Series of {1,3,4,9}
      detail columns: dict of pd.Series (0/1) for each test
    """
    s = pd.to_numeric(series, errors="coerce").copy()
    idx = s.index

    flag = pd.Series(1, index=idx, dtype="int16")
    missing = s.isna()
    flag[missing] = 9

    # Overrides
    ov = STATION_QC_OVERRIDES.get(station_code, {})
    gross_min = ov.get("gross_min_ft", STAGE_GROSS_MIN_FT)
    gross_max = ov.get("gross_max_ft", STAGE_GROSS_MAX_FT)

    # Gross range
    gross_fail = (~missing) & ((s < gross_min) | (s > gross_max))
    flag[gross_fail] = 4

    # Sentinel handling
    station_sentinels = ov.get("stage_sentinels", [])
    station_min_run = int(ov.get("sentinel_min_run", GLOBAL_STAGE_SENTINEL_MIN_RUN))

    global_sentinels = list(GLOBAL_STAGE_SENTINELS)
    sentinels = sorted(set(global_sentinels + station_sentinels))

    is_sentinel = (~missing) & s.isin(sentinels)
    rl = run_length(is_sentinel.fillna(False))
    sentinel_fail = is_sentinel & (rl >= station_min_run)

    # Special: for BCSA1299 we said min_run=1, so ANY -6 is fail
    flag[sentinel_fail] = 4

    # Flatline
    # Use diff threshold; constant sequences produce diff ~0
    dabs = s.diff().abs()
    flat = (~missing) & (dabs < STAGE_FLAT_EPS)
    flat_rl = run_length(flat.fillna(False))

    flat_sus = flat & (flat_rl >= STAGE_FLAT_SUSPECT_MIN_RUN) & (flat_rl < STAGE_FLAT_FAIL_MIN_RUN)
    flat_fail = flat & (flat_rl >= STAGE_FLAT_FAIL_MIN_RUN)

    # Don't override FAIL with SUSPECT
    flag[(flag == 1) & flat_sus] = 3
    flag[flat_fail] = 4

    # Rate of change
    # Evaluate one-step jump in ft per sample
    dd = s.diff().abs()
    roc_sus = (~missing) & (dd > STAGE_ROC_SUSPECT_FT) & (dd <= STAGE_ROC_FAIL_FT)
    roc_fail = (~missing) & (dd > STAGE_ROC_FAIL_FT)

    flag[(flag == 1) & roc_sus] = 3
    flag[roc_fail] = 4

    # Pad FAIL blocks so edges also fail (nuke-friendly)
    fail0 = (flag == 4)
    fail_pad = fail0.copy()
    for k in range(1, POST_PAD_SAMPLES + 1):
        fail_pad = fail_pad | fail0.shift(k) | fail0.shift(-k)
    fail_pad = fail_pad.fillna(False)
    flag[fail_pad] = 4

    detail = {
        "qartod_gross_fail": gross_fail.astype("int8"),
        "qartod_sentinel_fail": sentinel_fail.astype("int8"),
        "qartod_flat_sus": flat_sus.astype("int8"),
        "qartod_flat_fail": flat_fail.astype("int8"),
        "qartod_roc_sus": roc_sus.astype("int8"),
        "qartod_roc_fail": roc_fail.astype("int8"),
    }

    return flag, detail


def apply_qc_and_interpolate(series, flag_primary, kind):
    s = pd.to_numeric(series, errors="coerce").copy()

    # Always keep missing as NaN
    s[flag_primary == 9] = np.nan

    # Nuke FAIL (and any flags you choose)
    nuke_mask = pd.Series(False, index=s.index)
    for f in NUKE_QARTOD_FLAGS:
        nuke_mask = nuke_mask | (flag_primary == int(f))
    nuke_mask = nuke_mask.fillna(False)

    s[nuke_mask] = np.nan

    # Interpolate inside only
    if kind in ["stream", "weir", "wind", "rain"]:
        s = s.interpolate(
            method="time",
            limit=POST_INTERP_LIMIT,
            limit_direction="both",
            limit_area="inside"
        )

    # Clip for stage and rain
    if kind == "stream":
        s = s.clip(lower=STAGE_GROSS_MIN_FT, upper=STAGE_GROSS_MAX_FT)

    if kind == "rain":
        s = s.clip(lower=RAIN_GROSS_MIN_IN)

    # Aggressive final fill to eliminate NaNs before trim
    if kind in ["stream", "weir", "wind"]:
        before_na = int(s.isna().sum())
        if before_na > 0:
            print("[WARNING] [POST] Aggressive fill for", kind, "NaNs before:", before_na)
        s = s.ffill()
        s = s.bfill()
        after_na = int(s.isna().sum())
        if after_na > 0:
            print("[WARNING] [POST] Remaining NaNs after ffill/bfill for", kind, "count:", after_na, "forcing 0.0")
            s = s.fillna(0.0)

    if kind == "rain":
        before_na = int(s.isna().sum())
        if before_na > 0:
            print("[WARNING] [POST] Rain NaNs before fill:", before_na, "forcing 0.0")
        s = s.fillna(0.0)

    return s


# =============================================================================
# Time setup
# =============================================================================

START_TS = parse_utc(START_DATE_UTC, default_now=False)
END_TS = parse_utc(END_DATE_UTC, default_now=True)
CUTOFF_TS = parse_utc(CUTOFF_DATE_UTC, default_now=False)
SELECTED_CUTOFF_TS = parse_utc(SELECTED_STATIONS_CUTOFF_DATE_UTC, default_now=False)

if START_TS is not None:
    START_TS = START_TS - pd.Timedelta(hours=CUTOFF_RELAX_HOURS)
    print("[COMPLETED] START_TS relaxed by hours:", CUTOFF_RELAX_HOURS, "->", START_TS)

print("[COMPLETED] START_TS =", START_TS)
print("[COMPLETED] END_TS   =", END_TS)
print("[COMPLETED] CUTOFF_TS=", CUTOFF_TS)
print("[COMPLETED] SELECTED_CUTOFF_TS =", SELECTED_CUTOFF_TS)

# Pickle_Without_Trimming_simple.ipynb: download raw, clean, postprocess (inlined)
raw_records = []

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ----------------- APG SCADA -----------------
print("\n[COMPLETED] START Download APG SCADA")

all_targets = []
for fname, typ in APG_INDEX_FILES:
    index_url = apg_index_url_for(fname)
    print("[COMPLETED] [APG] Index:", index_url, "type:", typ, "| scheme:", APG_ID_SCHEME)

    r = request_with_retry(session, "GET", index_url, timeout=REQUEST_TIMEOUT)
    if r is None or r.status_code != 200:
        # FIX F4: FATAL, not `continue`. Skipping a catalogue used to silently drop every station of
        # that kind from the training matrix while the job still succeeded and uploaded a model.
        # request_with_retry has already retried RETRY_MAX times with backoff, so reaching here means
        # the catalogue is genuinely unreadable and we do not know what to download.
        raise RuntimeError(
            f"[APG][FATAL] Index request failed after retries: {index_url}. Refusing to build a "
            f"training matrix from an unknown station set (a partial matrix trains a model that has "
            f"never seen those gauges, silently)."
        )

    text = r.text
    buf = io.StringIO(text)
    codes = set()
    codes_loose = set()

    try:
        reader = csv.reader(buf)
        rows = list(reader)
        print("[COMPLETED] [APG] Index rows:", len(rows))
        for row in rows:
            for cell in row:
                for m in APG_CODE_RE.finditer(str(cell)):
                    codes.add(m.group(0))
                for m in APG_CODE_RE_LOOSE.finditer(str(cell)):
                    codes_loose.add(m.group(0))
    except Exception as exc:
        print("[WARNING] [APG] CSV parse failed, fallback regex on text:", repr(exc))
        for m in APG_CODE_RE.finditer(text):
            codes.add(m.group(0))
        for m in APG_CODE_RE_LOOSE.finditer(text):
            codes_loose.add(m.group(0))

    # Report anything the tightened pattern rejected, so a genuinely new code shape can never vanish
    # without a trace in the log.
    rejected = sorted(codes_loose - codes)
    if len(rejected) > 0:
        print("[WARNING] [APG] Candidates rejected by the strict code pattern:", rejected)
        print("[WARNING] [APG] If any of those is a REAL station, widen APG_CODE_RE deliberately.")

    print("[COMPLETED] [APG] Codes found:", len(codes))

    # FIX F4: per-catalogue floor.
    expected = APG_INDEX_EXPECTED_CODES.get(fname)
    if expected is not None:
        floor = max(1, expected - APG_INDEX_FLOOR_TOLERANCE)
        if len(codes) < floor:
            raise RuntimeError(
                f"[APG][FATAL] Catalogue {fname} yielded only {len(codes)} station code(s); "
                f"expected about {expected} (floor {floor}). This is the signature of APG changing "
                f"the catalogue format (e.g. dropping the legacy_name column). Refusing to train on "
                f"a silently truncated station set. Inspect {index_url} and update "
                f"APG_INDEX_EXPECTED_CODES / APG_CODE_RE deliberately."
            )
        if len(codes) > expected:
            print(f"[WARNING] [APG] {fname} now lists {len(codes)} codes, more than the expected "
                  f"{expected}. A NEW station changes the feature set and is a MODEL CHANGE -- "
                  f"review before it reaches a deployed model.")

    # Normalise APG's legacy_name to OUR station code before it becomes a filename and, downstream,
    # a training-matrix column name. Today this renames exactly one feed, BCRU4419 -> BCRU4420 (see
    # APG_LEGACY_NAME_TO_OUR_CODE); without it the matrix would gain a column the model has never
    # seen and lose one that three graph nodes depend on.
    normalised_codes = set()
    for code in sorted(codes):
        our_code = APG_LEGACY_NAME_TO_OUR_CODE.get(code, code)
        if our_code != code:
            print("[COMPLETED] [APG-ID] catalogue legacy_name", code, "-> our station code", our_code)
        normalised_codes.add(our_code)

    for code in sorted(normalised_codes):
        url = apg_data_url_for(code, typ)
        if url is None:
            # Under scheme='new' this means APG no longer publishes the station. Verified 2026-08-06:
            # MBSA4570 / MBRA4570 (APG-disregarded duplicate of MBSA5881, off the graph since
            # 2026-07-08) and MBRA8949 (retired ~2026-05-06, replaced by MBRA9909, its legacy
            # endpoint already 404s). Nothing still live is lost here.
            print("[WARNING] [APG-ID] No APG identifier for", code, typ,
                  "under scheme", APG_ID_SCHEME, "-- not downloaded.")
            continue
        all_targets.append((code, typ, url))

print("[COMPLETED] [APG] Total targets:", len(all_targets))

# FIX F4: total floor, catching a broad degradation that each per-file check might individually pass.
if len(all_targets) < APG_TOTAL_MIN_TARGETS:
    raise RuntimeError(
        f"[APG][FATAL] Only {len(all_targets)} APG download targets resolved; the floor is "
        f"{APG_TOTAL_MIN_TARGETS} (2026-08-05 baseline: 114). Refusing to build a training matrix "
        f"from a collapsed station set."
    )

for code, kind, url in all_targets:
    print("[COMPLETED] [APG] Download:", url)

    r = request_with_retry(session, "GET", url, stream=True, timeout=REQUEST_TIMEOUT)
    if r is None or r.status_code != 200:
        print("[UNSUCCESSFUL] [APG] GET failed:", url)
        continue

    try:
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception as exc:
        print("[ERROR] [APG] CSV read failed:", repr(exc))
        continue

    if df.shape[0] == 0:
        print("[WARNING] [APG] Empty file for", code, kind)
        continue

    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    candidates = [c for c in cols if str(c).lower() in ["timestamp", "time", "datetime", "date"]]
    if len(candidates) > 0:
        ts_col = candidates[0]
    else:
        ts_col = cols[0]

    val_cols = [c for c in cols if c != ts_col]
    if len(val_cols) == 0:
        print("[ERROR] [APG] No value column for", code, kind)
        continue

    val_col = val_cols[-1]

    df, idx_utc = parse_timestamp_column_to_utc(df, ts_col)
    vals = pd.to_numeric(df[val_col], errors="coerce")

    work = pd.DataFrame()
    work["timestamp_utc"] = idx_utc
    work["value_raw"] = vals.values

    work = work.dropna(subset=["timestamp_utc"])
    work = work.sort_values("timestamp_utc")
    work = work.drop_duplicates(subset=["timestamp_utc"], keep="last")

    if work.empty:
        print("[WARNING] [APG] No valid timestamps for", code, kind)
        continue

    validate_raw_frame(work, code + "_" + kind, kind)

    raw_path = os.path.join(RAW_DIR, code + APG_SUFFIX[kind])
    work.to_csv(raw_path, index=False)
    print("[COMPLETED] [APG] Successfully saved in:", raw_path)

    raw_records.append({
        "station": code,
        "kind": kind,
        "df": work.copy()
    })

    time.sleep(APG_SLEEP_BETWEEN)

# ----------------- USGS IV -----------------
print("\n[COMPLETED] START Download USGS IV")

for item in USGS_MAP:
    apg_filename = str(item["apg_filename"]).strip()
    kind = str(item["kind"]).strip().lower()
    site = str(item.get("site", "")).strip()
    series_idx = int(item.get("idx", 0))

    if site == "":
        print("[ERROR] [USGS] Missing site for", apg_filename)
        continue

    param = USGS_PARAM_FOR_KIND.get(kind)
    if param is None:
        print("[ERROR] [USGS] Unknown kind:", kind, "file", apg_filename)
        continue

    raw_path = os.path.join(USGS_RAW_DIR, apg_filename)

    existing_df = None
    existing_latest = None
    if os.path.isfile(raw_path):
        try:
            existing_df = pd.read_csv(raw_path)
        except Exception as exc:
            print("[WARNING] [USGS] Failed to read existing file:", raw_path, "error", repr(exc))
            existing_df = None

        if existing_df is not None:
            if "timestamp_utc" in existing_df.columns:
                ts_exist = pd.to_datetime(existing_df["timestamp_utc"], utc=True, errors="coerce")
            elif "timestamp_iso" in existing_df.columns:
                ts_exist = pd.to_datetime(existing_df["timestamp_iso"], utc=True, errors="coerce")
            else:
                ts_exist = pd.Series([], dtype="datetime64[ns, UTC]")
                print("[WARNING] [USGS] Existing file missing timestamp column:", raw_path)

            ts_exist = ts_exist.dropna()
            if len(ts_exist) > 0:
                existing_latest = ts_exist.max()
                print("[COMPLETED] [USGS] Existing latest timestamp for", apg_filename, ":", existing_latest)

    start_date = START_TS
    end_date = END_TS
    if existing_latest is not None:
        start_date = existing_latest

    if start_date is None or end_date is None:
        print("[ERROR] [USGS] Missing start or end date for", apg_filename)
        continue

    if start_date >= end_date:
        print("[COMPLETED] [USGS] No new data needed for", apg_filename)
        keep_existing_record(existing_df, apg_filename.replace(".csv", "").replace("_" + kind, ""), kind)
        continue

    print("[COMPLETED] [USGS] Download window for", apg_filename, "start", start_date, "end", end_date)

    frames = []
    cur = start_date
    while cur <= end_date:
        stop = min(cur + pd.Timedelta(days=USGS_CHUNK_DAYS), end_date)
        params = {
            "format": "json",
            "sites": site,
            "parameterCd": param,
            "startDT": cur.isoformat(),
            "endDT": stop.isoformat(),
            "siteStatus": "all",
        }

        r = request_with_retry(session, "GET", USGS_IV_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        if r is None or r.status_code != 200:
            print("[UNSUCCESSFUL] [USGS] Request failed:", apg_filename, "status", None if r is None else r.status_code)
            cur = stop + pd.Timedelta(seconds=1)
            time.sleep(USGS_PAUSE_SEC)
            continue

        try:
            payload = r.json()
        except Exception as exc:
            print("[ERROR] [USGS] JSON parse failed:", repr(exc))
            cur = stop + pd.Timedelta(seconds=1)
            time.sleep(USGS_PAUSE_SEC)
            continue

        container = payload.get("value", {})
        all_series = container.get("timeSeries", [])
        if len(all_series) == 0:
            cur = stop + pd.Timedelta(seconds=1)
            time.sleep(USGS_PAUSE_SEC)
            continue

        if series_idx < 0 or series_idx >= len(all_series):
            series_idx = 0

        series = all_series[series_idx]
        rows = []
        for block in series.get("values", []):
            for pt in block.get("value", []):
                t = pt.get("dateTime")
                v = pt.get("value")
                if t is None:
                    continue
                if v is None or v == "" or str(v).strip().lower() in ["ice"]:
                    continue
                rows.append((t, v))

        if len(rows) > 0:
            df_chunk = pd.DataFrame(rows, columns=["timestamp_iso", "value_raw"])
            df_chunk["timestamp_utc"] = pd.to_datetime(df_chunk["timestamp_iso"], utc=True, errors="coerce")
            df_chunk["value_raw"] = pd.to_numeric(df_chunk["value_raw"], errors="coerce")
            df_chunk = df_chunk.dropna(subset=["timestamp_utc", "value_raw"])
            df_chunk = df_chunk.sort_values("timestamp_utc")
            frames.append(df_chunk[["timestamp_utc", "value_raw"]])

        cur = stop + pd.Timedelta(seconds=1)
        time.sleep(USGS_PAUSE_SEC)

    df_new = None
    if len(frames) > 0:
        df_new = pd.concat(frames, ignore_index=True)
        df_new = df_new.drop_duplicates(subset=["timestamp_utc"], keep="last")
        df_new = df_new.sort_values("timestamp_utc")

    df_old = None
    if existing_df is not None:
        if "timestamp_utc" in existing_df.columns and "value_raw" in existing_df.columns:
            df_old = existing_df[["timestamp_utc", "value_raw"]].copy()
            df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_utc"], utc=True, errors="coerce")
            df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
            df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
        elif "timestamp_iso" in existing_df.columns and "value_raw" in existing_df.columns:
            df_old = existing_df.copy()
            df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_iso"], utc=True, errors="coerce")
            df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
            df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
            df_old = df_old[["timestamp_utc", "value_raw"]]

    if df_new is None and df_old is None:
        print("[UNSUCCESSFUL] [USGS] No data available for", apg_filename)
        continue
    if df_new is None and df_old is not None:
        df_all = df_old.copy()
    elif df_old is None and df_new is not None:
        df_all = df_new.copy()
    else:
        df_all = pd.concat([df_old, df_new], ignore_index=True)

    df_all = df_all.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df_all = df_all.sort_values("timestamp_utc")

    validate_raw_frame(df_all, apg_filename, kind)

    df_all.to_csv(raw_path, index=False)
    print("[COMPLETED] [USGS] Successfully saved in:", raw_path)

    station_code = apg_filename.replace(".csv", "").replace("_" + kind, "")
    raw_records.append({"station": station_code, "kind": kind, "df": df_all.copy()})

# ----------------- NOAA ASOS -----------------
print("\n[COMPLETED] START Download NOAA ASOS")

for station_id in ASOS_STATION_IDS:
    if station_id not in ASOS_STATION_MAP:
        print("[ERROR] [ASOS] No mapping for station", station_id)
        continue

    wind_code = ASOS_STATION_MAP[station_id]["wind_code"]
    rain_code = ASOS_STATION_MAP[station_id]["rain_code"]
    asos_id = ASOS_STATION_MAP[station_id].get("asos_id", station_id)

    wind_path = os.path.join(NOAA_RAW_DIR, wind_code + "_wind.csv")
    rain_path = os.path.join(NOAA_RAW_DIR, rain_code + "_rain.csv") if rain_code else None

    wind_existing = None
    rain_existing = None
    wind_latest = None
    rain_latest = None

    if os.path.isfile(wind_path):
        try:
            wind_existing = pd.read_csv(wind_path)
        except Exception as exc:
            print("[WARNING] [ASOS] Failed to read existing wind file:", wind_path, "error", repr(exc))
            wind_existing = None
        if wind_existing is not None and "timestamp_utc" in wind_existing.columns:
            ts_wind = pd.to_datetime(wind_existing["timestamp_utc"], utc=True, errors="coerce")
            ts_wind = ts_wind.dropna()
            if len(ts_wind) > 0:
                wind_latest = ts_wind.max()

    if rain_path is not None and os.path.isfile(rain_path):
        try:
            rain_existing = pd.read_csv(rain_path)
        except Exception as exc:
            print("[WARNING] [ASOS] Failed to read existing rain file:", rain_path, "error", repr(exc))
            rain_existing = None
        if rain_existing is not None and "timestamp_utc" in rain_existing.columns:
            ts_rain = pd.to_datetime(rain_existing["timestamp_utc"], utc=True, errors="coerce")
            ts_rain = ts_rain.dropna()
            if len(ts_rain) > 0:
                rain_latest = ts_rain.max()

    start_date = START_TS
    end_date = END_TS

    if wind_latest is not None and rain_latest is not None:
        start_date = min(wind_latest, rain_latest)
    elif wind_latest is not None:
        start_date = wind_latest
    elif rain_latest is not None:
        start_date = rain_latest

    if start_date is None or end_date is None:
        print("[ERROR] [ASOS] Missing start or end date for", station_id)
        continue

    if start_date >= end_date:
        print("[COMPLETED] [ASOS] No new data needed for", station_id)
        keep_existing_record(wind_existing, wind_code, "wind")
        if rain_code:
            keep_existing_record(rain_existing, rain_code, "rain")
        continue

    print("[COMPLETED] [ASOS] Download window for", station_id, "start", start_date, "end", end_date)

    all_chunks = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + pd.Timedelta(days=365), end_date)
        params = {
            "station": asos_id,
            "data": ["drct", "sknt", "gust", "p01i"],
            "year1": cur.year,
            "month1": cur.month,
            "day1": cur.day,
            "year2": chunk_end.year,
            "month2": chunk_end.month,
            "day2": chunk_end.day,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "elev": "no",
            "direct": "no",
            "report_type": ["3", "4"],
        }

        r = request_with_retry(session, "GET", ASOS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r is None or r.status_code != 200:
            print("[UNSUCCESSFUL] [ASOS] Request failed:", station_id, "status", None if r is None else r.status_code)
            cur = chunk_end + pd.Timedelta(days=1)
            continue

        lines = []
        for line in r.text.splitlines():
            if line.strip() == "":
                continue
            if line.startswith("#"):
                continue
            lines.append(line)

        if len(lines) == 0:
            cur = chunk_end + pd.Timedelta(days=1)
            continue

        cleaned_text = "\n".join(lines)
        df_chunk = pd.read_csv(io.StringIO(cleaned_text))
        df_chunk.columns = [c.strip().lower() for c in df_chunk.columns]

        if "valid" not in df_chunk.columns:
            cur = chunk_end + pd.Timedelta(days=1)
            continue

        df_chunk["valid"] = pd.to_datetime(df_chunk["valid"], errors="coerce", utc=True)
        df_chunk = df_chunk[df_chunk["valid"].notna()]

        all_chunks.append(df_chunk)
        cur = chunk_end + pd.Timedelta(days=1)

    if len(all_chunks) == 0:
        print("[WARNING] [ASOS] No new data for", station_id)
        continue

    df_all = pd.concat(all_chunks, ignore_index=True)
    df_all = df_all.sort_values("valid")
    df_all = df_all.drop_duplicates(subset=["valid"], keep="last")

    df_all["sknt"] = pd.to_numeric(df_all.get("sknt"), errors="coerce")
    df_all["p01i"] = pd.to_numeric(df_all.get("p01i"), errors="coerce")

    if "sknt" in df_all.columns:
        df_wind = pd.DataFrame()
        df_wind["timestamp_utc"] = df_all["valid"]
        df_wind["value_raw"] = df_all["sknt"] * 1.150779448
        df_wind = df_wind.dropna(subset=["timestamp_utc", "value_raw"])
        df_wind = df_wind.sort_values("timestamp_utc")

        if wind_existing is not None and "timestamp_utc" in wind_existing.columns and "value_raw" in wind_existing.columns:
            df_old = wind_existing[["timestamp_utc", "value_raw"]].copy()
            df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_utc"], utc=True, errors="coerce")
            df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
            df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
            df_wind = pd.concat([df_old, df_wind], ignore_index=True)

        df_wind = df_wind.drop_duplicates(subset=["timestamp_utc"], keep="last")
        df_wind = df_wind.sort_values("timestamp_utc")
        validate_raw_frame(df_wind, wind_code + "_wind", "wind")

        df_wind.to_csv(wind_path, index=False)
        print("[COMPLETED] [ASOS] Successfully saved in:", wind_path)
        raw_records.append({"station": wind_code, "kind": "wind", "df": df_wind.copy()})
    else:
        print("[UNSUCCESSFUL] [ASOS] Wind data missing for", station_id)

    if rain_code and "p01i" in df_all.columns:
        df_rain = pd.DataFrame()
        df_rain["timestamp_utc"] = df_all["valid"]
        df_rain["value_raw"] = df_all["p01i"]
        df_rain = df_rain.dropna(subset=["timestamp_utc", "value_raw"])
        df_rain = df_rain.sort_values("timestamp_utc")

        if rain_existing is not None and "timestamp_utc" in rain_existing.columns and "value_raw" in rain_existing.columns:
            df_old = rain_existing[["timestamp_utc", "value_raw"]].copy()
            df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_utc"], utc=True, errors="coerce")
            df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
            df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
            df_rain = pd.concat([df_old, df_rain], ignore_index=True)

        df_rain = df_rain.drop_duplicates(subset=["timestamp_utc"], keep="last")
        df_rain = df_rain.sort_values("timestamp_utc")
        validate_raw_frame(df_rain, rain_code + "_rain", "rain")

        df_rain.to_csv(rain_path, index=False)
        print("[COMPLETED] [ASOS] Successfully saved in:", rain_path)
        raw_records.append({"station": rain_code, "kind": "rain", "df": df_rain.copy()})
    else:
        print("[UNSUCCESSFUL] [ASOS] Rain data missing for", station_id)

# ----------------- NOAA COOPS -----------------
print("\n[COMPLETED] START Download NOAA COOPS")

end_date = END_TS

# WIND
wind_path = os.path.join(NOAA_RAW_DIR, COOPS_WIND_CODE + "_wind.csv")
wind_existing = None
wind_latest = None

if os.path.isfile(wind_path):
    try:
        wind_existing = pd.read_csv(wind_path)
    except Exception as exc:
        print("[WARNING] [COOPS] Failed to read existing wind file:", wind_path, "error", repr(exc))
        wind_existing = None

    if wind_existing is not None and "timestamp_utc" in wind_existing.columns:
        ts_wind = pd.to_datetime(wind_existing["timestamp_utc"], utc=True, errors="coerce")
        ts_wind = ts_wind.dropna()
        if len(ts_wind) > 0:
            wind_latest = ts_wind.max()

wind_start = START_TS
if wind_latest is not None:
    wind_start = wind_latest

if wind_start is not None and end_date is not None:
    if wind_start >= end_date:
        print("[COMPLETED] [COOPS] No new wind data needed")
        keep_existing_record(wind_existing, COOPS_WIND_CODE, "wind")
    else:
        wind_rows = []
        cur = wind_start
        while cur <= end_date:
            stop = min(cur + pd.Timedelta(days=COOPS_MAX_DAYS), end_date)
            params = {
                "product": "wind",
                "application": "mfg_python_script",
                "station": COOPS_STATION_ID,
                "begin_date": cur.strftime("%Y%m%d"),
                "end_date": stop.strftime("%Y%m%d"),
                "time_zone": "gmt",
                "units": "english",
                "format": "json",
            }

            r = request_with_retry(session, "GET", COOPS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r is None or r.status_code != 200:
                print("[UNSUCCESSFUL] [COOPS] Wind request failed status", None if r is None else r.status_code)
                cur = stop + pd.Timedelta(days=1)
                continue

            try:
                js = r.json()
            except Exception as exc:
                print("[ERROR] [COOPS] Wind JSON parse failed:", repr(exc))
                cur = stop + pd.Timedelta(days=1)
                continue

            data = js.get("data", [])
            for row in data:
                t = row.get("t")
                spd = row.get("s")
                if t is None or spd is None:
                    continue
                wind_rows.append((t, spd))

            cur = stop + pd.Timedelta(days=1)

        if len(wind_rows) > 0:
            df_wind = pd.DataFrame(wind_rows, columns=["timestamp_iso", "speed_knots"])
            df_wind["timestamp_utc"] = pd.to_datetime(df_wind["timestamp_iso"], utc=True, errors="coerce")
            df_wind["value_raw"] = pd.to_numeric(df_wind["speed_knots"], errors="coerce") * 1.150779448
            df_wind = df_wind.dropna(subset=["timestamp_utc", "value_raw"])
            df_wind = df_wind.sort_values("timestamp_utc")

            if wind_existing is not None and "timestamp_utc" in wind_existing.columns and "value_raw" in wind_existing.columns:
                df_old = wind_existing[["timestamp_utc", "value_raw"]].copy()
                df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_utc"], utc=True, errors="coerce")
                df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
                df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
                df_wind = pd.concat([df_old, df_wind], ignore_index=True)

            df_wind = df_wind.drop_duplicates(subset=["timestamp_utc"], keep="last")
            df_wind = df_wind.sort_values("timestamp_utc")
            validate_raw_frame(df_wind, COOPS_WIND_CODE + "_wind", "wind")

            df_wind.to_csv(wind_path, index=False)
            print("[COMPLETED] [COOPS] Successfully saved in:", wind_path)
            raw_records.append({"station": COOPS_WIND_CODE, "kind": "wind", "df": df_wind.copy()})
        else:
            print("[WARNING] [COOPS] No new wind data")

# WATER LEVEL
stream_path = os.path.join(NOAA_RAW_DIR, COOPS_STREAM_CODE + "_stream.csv")
stream_existing = None
stream_latest = None

if os.path.isfile(stream_path):
    try:
        stream_existing = pd.read_csv(stream_path)
    except Exception as exc:
        print("[WARNING] [COOPS] Failed to read existing stream file:", stream_path, "error", repr(exc))
        stream_existing = None

    if stream_existing is not None and "timestamp_utc" in stream_existing.columns:
        ts_stream = pd.to_datetime(stream_existing["timestamp_utc"], utc=True, errors="coerce")
        ts_stream = ts_stream.dropna()
        if len(ts_stream) > 0:
            stream_latest = ts_stream.max()

stream_start = START_TS
if stream_latest is not None:
    stream_start = stream_latest

if stream_start is not None and end_date is not None:
    if stream_start >= end_date:
        print("[COMPLETED] [COOPS] No new stream data needed")
        keep_existing_record(stream_existing, COOPS_STREAM_CODE, "stream")
    else:
        wl_rows = []
        cur = stream_start
        while cur <= end_date:
            stop = min(cur + pd.Timedelta(days=COOPS_MAX_DAYS), end_date)
            params = {
                "product": "water_level",
                "application": "mfg_python_script",
                "station": COOPS_STATION_ID,
                "begin_date": cur.strftime("%Y%m%d"),
                "end_date": stop.strftime("%Y%m%d"),
                "time_zone": "gmt",
                "units": "english",
                "format": "json",
                "datum": "STND",
            }

            r = request_with_retry(session, "GET", COOPS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if r is None or r.status_code != 200:
                print("[UNSUCCESSFUL] [COOPS] Stream request failed status", None if r is None else r.status_code)
                cur = stop + pd.Timedelta(days=1)
                continue

            try:
                js = r.json()
            except Exception as exc:
                print("[ERROR] [COOPS] Stream JSON parse failed:", repr(exc))
                cur = stop + pd.Timedelta(days=1)
                continue

            data = js.get("data", [])
            for row in data:
                t = row.get("t")
                v = row.get("v")
                if t is None or v is None:
                    continue
                wl_rows.append((t, v))

            cur = stop + pd.Timedelta(days=1)

        if len(wl_rows) > 0:
            df_wl = pd.DataFrame(wl_rows, columns=["timestamp_iso", "water_ft"])
            df_wl["timestamp_utc"] = pd.to_datetime(df_wl["timestamp_iso"], utc=True, errors="coerce")
            df_wl["value_raw"] = pd.to_numeric(df_wl["water_ft"], errors="coerce") + COOPS_NAVD_OFFSET_FT
            df_wl = df_wl.dropna(subset=["timestamp_utc", "value_raw"])
            df_wl = df_wl.sort_values("timestamp_utc")

            if stream_existing is not None and "timestamp_utc" in stream_existing.columns and "value_raw" in stream_existing.columns:
                df_old = stream_existing[["timestamp_utc", "value_raw"]].copy()
                df_old["timestamp_utc"] = pd.to_datetime(df_old["timestamp_utc"], utc=True, errors="coerce")
                df_old["value_raw"] = pd.to_numeric(df_old["value_raw"], errors="coerce")
                df_old = df_old.dropna(subset=["timestamp_utc", "value_raw"])
                df_wl = pd.concat([df_old, df_wl], ignore_index=True)

            df_wl = df_wl.drop_duplicates(subset=["timestamp_utc"], keep="last")
            df_wl = df_wl.sort_values("timestamp_utc")
            validate_raw_frame(df_wl, COOPS_STREAM_CODE + "_stream", "stream")

            df_wl.to_csv(stream_path, index=False)
            print("[COMPLETED] [COOPS] Successfully saved in:", stream_path)
            raw_records.append({"station": COOPS_STREAM_CODE, "kind": "stream", "df": df_wl.copy()})
        else:
            print("[WARNING] [COOPS] No new stream data")

# NOTE: the pre-clean USGS datum pass was disabled here; the real datum
# correction runs unconditionally in the postprocess section further below.
print("\n[COMPLETED] SKIP USGS DATUM adder (ADD_USGS_DATUM=False)")

print("[COMPLETED] raw_records total:", len(raw_records))

# =============================================================================
# Clean RAW -> Cleaned
# =============================================================================

print("\n[COMPLETED] START Clean RAW to Cleaned")

raw_files_by_name = {}
for directory in CLEAN_INPUT_DIRS:
    if not os.path.isdir(directory):
        print("[WARNING] [CLEAN] Input dir missing:", directory)
        continue

    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if len(files) == 0:
        print("[WARNING] [CLEAN] No CSV files in:", directory)

    for path in files:
        name = os.path.basename(path)
        raw_files_by_name[name] = path

if len(raw_files_by_name) == 0:
    raise RuntimeError("[CLEAN] No raw files found")

cleaned_count = 0
skipped_count = 0

for name in sorted(raw_files_by_name.keys()):
    path = raw_files_by_name[name]
    kind = detect_kind_from_filename(path)
    if kind is None:
        print("[WARNING] [CLEAN] Unknown file type:", path)
        skipped_count = skipped_count + 1
        continue

    print("[COMPLETED] [CLEAN] Reading raw:", path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print("[ERROR] [CLEAN] Read failed:", path, "error", repr(exc))
        skipped_count = skipped_count + 1
        continue

    if df.empty:
        print("[WARNING] [CLEAN] Empty raw file:", path)
        skipped_count = skipped_count + 1
        continue

    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    if "timestamp_utc" in lower_map:
        ts_col = lower_map["timestamp_utc"]
    elif "timestamp" in lower_map:
        ts_col = lower_map["timestamp"]
    elif "time" in lower_map:
        ts_col = lower_map["time"]
    elif "datetime" in lower_map:
        ts_col = lower_map["datetime"]
    elif "date" in lower_map:
        ts_col = lower_map["date"]
    else:
        ts_col = cols[0]

    other_cols = [c for c in df.columns if c != ts_col]
    if len(other_cols) == 0:
        print("[ERROR] [CLEAN] No value column in:", path)
        skipped_count = skipped_count + 1
        continue

    value_candidates = []
    for c in other_cols:
        c_low = str(c).lower()
        if c_low in ["value_raw", "value", "stage_ft", "rain_in", "wind_mph", "water_ft", "status"]:
            value_candidates.append(c)

    if len(value_candidates) == 0:
        value_candidates = list(other_cols)

    best_col = None
    best_non_null = -1
    for c in value_candidates:
        s_tmp = pd.to_numeric(df[c], errors="coerce")
        non_null = int(s_tmp.notna().sum())
        print("[COMPLETED] [CLEAN] Candidate value column:", c, "non_null=", non_null)
        if non_null > best_non_null:
            best_non_null = non_null
            best_col = c

    if best_col is None:
        print("[ERROR] [CLEAN] No numeric value column in:", path)
        skipped_count = skipped_count + 1
        continue

    val_col = best_col
    print("[COMPLETED] [CLEAN] Selected value column:", val_col)

    df["timestamp_raw"] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
    bad_ts = int(df["timestamp_raw"].isna().sum())
    if bad_ts > 0:
        print("[WARNING] [CLEAN] Unparsable timestamps:", bad_ts, "file", path)

    df = df.dropna(subset=["timestamp_raw"])
    if df.empty:
        print("[UNSUCCESSFUL] [CLEAN] No valid timestamps after parse:", path)
        skipped_count = skipped_count + 1
        continue

    idx = pd.DatetimeIndex(df["timestamp_raw"])
    tz_present = pd.api.types.is_datetime64tz_dtype(idx)

    if not tz_present:
        try:
            idx_local = idx.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
        except Exception as exc:
            print("[WARNING] [CLEAN] tz_localize infer failed:", repr(exc))
            idx_local = idx.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
            mask_nat = idx_local.isna()
            if mask_nat.any():
                df = df.loc[~mask_nat].copy()
                idx_local = idx_local[~mask_nat]
        idx_utc = idx_local.tz_convert("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    work = df[["value"]].copy()
    work.index = idx_utc
    work = work.sort_index()

    dup_count = int(work.index.duplicated().sum())
    if dup_count > 0:
        print("[WARNING] [CLEAN] Duplicate timestamps:", dup_count, "file", path)
        work = work[~work.index.duplicated(keep="last")]

    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")

    if work.empty:
        print("[UNSUCCESSFUL] [CLEAN] Empty after cleanup:", path)
        skipped_count = skipped_count + 1
        continue

    if kind == "rain":
        s = work["value"].resample(RESAMPLE_RULE).sum(min_count=1)
        neg = s < CLEAN_SANITY["rain_in_min_per_15m"]
        if neg.any():
            print("[WARNING] [CLEAN] Rain negatives:", int(neg.sum()))
            s[neg] = 0.0
        too_high = s > CLEAN_SANITY["rain_in_max_per_15m"]
        if too_high.any():
            print("[WARNING] [CLEAN] Rain spikes:", int(too_high.sum()))
            s[too_high] = np.nan
        s = s.fillna(0.0)
        df_clean = s.to_frame(CLEAN_COLMAP[kind])

    elif kind in ["stream", "weir"]:
        s = work["value"].resample(RESAMPLE_RULE).last()
        if kind == "stream":
            low = s < CLEAN_SANITY["stage_ft_min"]
            high = s > CLEAN_SANITY["stage_ft_max"]
        else:
            low = s < CLEAN_SANITY["weir_ft_min"]
            high = s > CLEAN_SANITY["weir_ft_max"]

        if low.any() or high.any():
            print("[WARNING] [CLEAN]", kind, "out-of-range low", int(low.sum()), "high", int(high.sum()))
            s[low | high] = np.nan

        s = s.interpolate(method="time", limit_direction="both")
        s = s.ffill()
        s = s.bfill()
        if s.isna().sum() > 0:
            print("[WARNING] [CLEAN]", kind, "NaN remained, filling 0.0")
            s = s.fillna(0.0)

        df_clean = s.to_frame(CLEAN_COLMAP[kind])

    elif kind in ["pump", "gate"]:
        s = work["value"].resample(RESAMPLE_RULE).last()
        s = s.round().astype("float")

        # ---- STRUCTURE QC + GATE STATUS DECODE (work item W1, 2026-08-06) ----
        # Both run BEFORE the isin([0,1]) check below. Order is not cosmetic: that check sends
        # Status 2 (Closed) and 3 (Error) to NaN, and the ffill underneath then carries the previous
        # state forward -- so a gate that closes keeps training the model as OPEN.
        if STRUCT_QC_ENABLE:
            valid_values = GATE_VALID_STATUS_VALUES if kind == "gate" else PUMP_VALID_STATUS_VALUES
            bad_mask = s.notna() & (~s.isin(list(valid_values)))
            if bool(bad_mask.any()):
                print("[WARNING] [QC][STRUCT] out-of-domain", kind, "value(s):", int(bad_mask.sum()),
                      "| examples", sorted(set(s[bad_mask].tolist()))[:5],
                      "-> masked to NaN (valid domain", list(valid_values), ")")
                s = s.mask(bad_mask)

        if kind == "gate" and GATE_STATUS_DECODE_ENABLE:
            n_closed = int((s == 2.0).sum())
            n_error = int((s == 3.0).sum())
            s = s.map(GATE_STATUS_DECODE)
            if n_closed or n_error:
                print("[COMPLETED] [STRUCT][GATE-DECODE] Status=2 (Closed) ->0:", n_closed,
                      "| Status=3 (Error) ->NaN:", n_error)

        invalid = ~s.isin([0.0, 1.0])
        if invalid.any():
            print("[WARNING] [CLEAN] Invalid", kind, "states:", int(invalid.sum()))
            s[invalid] = np.nan

        s = s.ffill()
        s = s.bfill()
        if s.isna().all():
            print("[WARNING] [CLEAN] No valid", kind, "states. Forcing zeros.")
            s = s.fillna(0.0)

        df_clean = s.astype("Int64").to_frame(CLEAN_COLMAP[kind])

    elif kind == "wind":
        s = work["value"].resample(RESAMPLE_RULE).mean()
        neg = s < CLEAN_SANITY["wind_mph_min"]
        high = s > CLEAN_SANITY["wind_mph_max"]
        if neg.any() or high.any():
            print("[WARNING] [CLEAN] Wind out-of-range:", int((neg | high).sum()))
            s[neg | high] = np.nan

        s = s.interpolate(method="time", limit_direction="both")
        s = s.ffill()
        s = s.bfill()
        if s.isna().sum() > 0:
            print("[WARNING] [CLEAN] Wind NaN remained, filling 0.0")
            s = s.fillna(0.0)

        df_clean = s.to_frame(CLEAN_COLMAP[kind])

    else:
        print("[UNSUCCESSFUL] [CLEAN] Unknown kind:", kind)
        skipped_count = skipped_count + 1
        continue

    df_clean = df_clean.copy()
    df_clean["timestamp_utc"] = df_clean.index
    df_clean["timestamp_local"] = df_clean.index.tz_convert(LOCAL_TZ)

    value_col = CLEAN_COLMAP[kind]
    df_clean = df_clean[["timestamp_utc", "timestamp_local", value_col]]

    if df_clean[value_col].isna().sum() > 0:
        print("[WARNING] [CLEAN] NaN remained in", value_col, "forcing fill")
        if kind in ["pump", "gate"]:
            df_clean[value_col] = df_clean[value_col].fillna(0).astype("Int64")
        else:
            df_clean[value_col] = df_clean[value_col].fillna(0.0)

    out_path = os.path.join(CLEAN_DIR, os.path.basename(path))
    df_clean.to_csv(out_path, index=False)
    print("[COMPLETED] [CLEAN] Successfully saved in:", out_path)

    cleaned_count = cleaned_count + 1

print("[COMPLETED] [CLEAN] Files written:", cleaned_count)
print("[COMPLETED] [CLEAN] Files skipped:", skipped_count)

# =============================================================================
# Post-Processing (QARTOD-style + Nuke -6 for BCSA1299) + FINAL NO-NaN FILL (BEFORE TRIM)
# =============================================================================

print("\n[COMPLETED] START Postprocess Cleaned (FINAL NO-NaN FILL happens here, before TRIM)")

clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.csv")))
if len(clean_files) == 0:
    raise RuntimeError("[POST] No cleaned files found")

post_count = 0

for path in clean_files:
    base = os.path.basename(path)
    station_code = base.split("_")[0]
    kind = detect_kind_from_filename(path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print("[ERROR] [POST] Read failed:", path, "error", repr(exc))
        continue

    if "timestamp_utc" not in df.columns:
        print("[UNSUCCESSFUL] [POST] Missing timestamp_utc:", path)
        continue

    t = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    valid_mask = t.notna()
    if not valid_mask.any():
        print("[WARNING] [POST] All timestamps invalid:", path)
        continue

    df = df.loc[valid_mask].copy()
    t = t.loc[valid_mask]
    df.index = t
    df = df.sort_index()

    # -------------------------------------------------------------
    # Stage QARTOD (stream)
    # -------------------------------------------------------------
    if kind == "stream" and "stage_ft" in df.columns:
        ser0 = pd.to_numeric(df["stage_ft"], errors="coerce")

        print("[COMPLETED] [POST] Stage QARTOD:", base)
        flag_primary, detail = qartod_primary_stage(ser0, station_code)

        # Nuke FAIL and interpolate inside (your existing logic)
        ser1 = apply_qc_and_interpolate(ser0, flag_primary, "stream")

        # Save columns
        df["stage_ft_qartod_flag"] = flag_primary.values.astype(np.int16)

        for k in detail:
            df[k] = detail[k].values.astype(np.int8)

        df["stage_ft_raw"] = ser0.values.astype("float32")
        df["stage_ft"] = ser1.values.astype("float32")

        n_fail = int((flag_primary == 4).sum())
        n_sus = int((flag_primary == 3).sum())
        n_miss = int((flag_primary == 9).sum())
        print("[COMPLETED] [POST] Stage flags counts: FAIL", n_fail, "SUS", n_sus, "MISS", n_miss)

        # Plot
        if POSTPROCESS_PLOTS:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, ser0, alpha=0.6, label="Original")
            fail_mask = (flag_primary == 4).reindex(df.index, fill_value=False)
            if fail_mask.any():
                bad_idx = df.index[fail_mask]
                plt.scatter(bad_idx, ser0.loc[bad_idx], s=12, label="FAIL (nuked)")
            plt.plot(df.index, ser1, alpha=0.9, linestyle="--", label="Cleaned")
            plt.title("Stage QARTOD cleaned - " + base)
            plt.xlabel("Time (UTC)")
            plt.ylabel("stage_ft")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(POSTPROCESSED_PLOT_DIR, base.replace(".csv", "_stage_qartod.png"))
            plt.savefig(plot_path)
            plt.close()
            print("[COMPLETED] [POST] Stage plot saved:", plot_path)

    # -------------------------------------------------------------
    # Rain QARTOD-lite (gross range + nuke)
    # -------------------------------------------------------------
    if kind == "rain" and "rain_in" in df.columns:
        ser0 = pd.to_numeric(df["rain_in"], errors="coerce")

        flag = pd.Series(1, index=df.index, dtype="int16")
        miss = ser0.isna()
        flag[miss] = 9

        gross_fail = (~miss) & ((ser0 < RAIN_GROSS_MIN_IN) | (ser0 > RAIN_GROSS_MAX_IN))
        flag[gross_fail] = 4

        ser1 = ser0.copy()
        ser1[flag == 4] = np.nan
        ser1 = ser1.interpolate(method="time", limit=RAIN_INTERP_LIMIT, limit_direction="both", limit_area="inside")
        ser1 = ser1.clip(lower=RAIN_GROSS_MIN_IN)
        ser1 = ser1.fillna(0.0)

        df["rain_in_qartod_flag"] = flag.values.astype(np.int16)
        df["rain_in_raw"] = ser0.values.astype("float32")
        df["rain_in"] = ser1.values.astype("float32")

        if POSTPROCESS_PLOTS:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, ser0, alpha=0.6, label="Original")
            if gross_fail.any():
                bad_idx = df.index[gross_fail]
                plt.scatter(bad_idx, ser0.loc[bad_idx], s=12, label="FAIL (nuked)")
            plt.plot(df.index, ser1, alpha=0.9, linestyle="--", label="Cleaned")
            plt.title("Rain cleaned - " + base)
            plt.xlabel("Time (UTC)")
            plt.ylabel("rain_in")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(POSTPROCESSED_PLOT_DIR, base.replace(".csv", "_rain_qartod.png"))
            plt.savefig(plot_path)
            plt.close()
            print("[COMPLETED] [POST] Rain plot saved:", plot_path)

    # -------------------------------------------------------------
    # FINAL NO-NaN FILL (this is the key: BEFORE TRIM)
    # Also write a mask for what was originally missing
    # -------------------------------------------------------------
    fill_targets = []

    if kind == "stream" and "stage_ft" in df.columns:
        fill_targets.append("stage_ft")

    if kind == "rain" and "rain_in" in df.columns:
        fill_targets.append("rain_in")

    if kind == "wind" and "wind_mph" in df.columns:
        fill_targets.append("wind_mph")

    if kind == "weir" and "elev_ft" in df.columns:
        fill_targets.append("elev_ft")

    if kind in ["pump", "gate"] and "status" in df.columns:
        fill_targets.append("status")

    if len(fill_targets) > 0:
        for vcol in fill_targets:
            nan_before = int(df[vcol].isna().sum())
            if nan_before > 0:
                print("[FINALFILL] Found NaNs for", base, "col", vcol, "count", nan_before)

            # Save missingness mask per file+col
            mask_name = base.replace(".csv", "") + "__" + vcol + "__nanmask.csv"
            mask_path = os.path.join(RUN_QC_DIR, mask_name)

            nanmask = df[vcol].isna().astype("int8").to_frame(vcol + "_nanmask")
            nanmask["timestamp_utc"] = df.index
            nanmask = nanmask[["timestamp_utc", vcol + "_nanmask"]]
            nanmask.to_csv(mask_path, index=False)
            print("[FINALFILL] Wrote nanmask:", mask_path)

            # Fill rules by variable type
            if vcol == "rain_in":
                # Missing rain -> 0
                df[vcol] = df[vcol].fillna(0.0)

            elif vcol == "status":
                # Status: carry forward, then backward, then force 0
                df[vcol] = df[vcol].ffill()
                df[vcol] = df[vcol].bfill()
                df[vcol] = df[vcol].fillna(0.0)
                df[vcol] = df[vcol].round().clip(lower=0, upper=1)

            else:
                # Continuous: interpolate using time index, then edge fills
                df[vcol] = df[vcol].interpolate(method="time", limit_direction="both")
                df[vcol] = df[vcol].ffill()
                df[vcol] = df[vcol].bfill()

                # Absolute guarantee if still NaN (all-NaN column case)
                if int(df[vcol].isna().sum()) > 0:
                    med = float(pd.to_numeric(df[vcol], errors="coerce").median(skipna=True))
                    if np.isnan(med):
                        med = 0.0
                    df[vcol] = df[vcol].fillna(med)
                    df[vcol] = df[vcol].fillna(0.0)

            nan_after = int(df[vcol].isna().sum())
            print("[FINALFILL] After fill", base, "col", vcol, "NaNs =", nan_after)

    # -------------------------------------------------------------
    # Write postprocessed output (already final-filled, so TRIM sees no NaNs)
    # -------------------------------------------------------------
    out_path = os.path.join(POSTPROCESSED_DIR, base)
    df_out = df.reset_index(drop=True).copy()
    df_out.to_csv(out_path, index=False)
    print("[COMPLETED] [POST] Successfully saved in:", out_path)

    post_count = post_count + 1

print("[COMPLETED] [POST] Files written:", post_count)

print("\n[COMPLETED] START USGS DATUM adder")

# Build stream mapping (file_stem, site_no, index) from the stream rows of USGS_MAP.
usgs_stream_mapping = []
_seen = set()
for item in USGS_MAP:
    kind = str(item.get("kind", "")).strip().lower()
    if kind != "stream":
        continue
    apg_filename = str(item.get("apg_filename", "")).strip()
    site_no = str(item.get("site", "")).strip()
    idx = int(item.get("idx", 0))
    if apg_filename == "" or site_no == "":
        print("[WARNING] [USGS DATUM] bad stream mapping row:", item)
        continue
    if not apg_filename.lower().endswith("_stream.csv"):
        print("[WARNING] [USGS DATUM] not a stream filename:", apg_filename)
        continue
    file_stem = apg_filename[:-len("_stream.csv")]
    key = (file_stem, site_no, idx)
    if key in _seen:
        continue
    _seen.add(key)
    usgs_stream_mapping.append({"file_stem": file_stem, "site_no": site_no, "index": idx, "apg_filename": apg_filename})
print("[COMPLETED] [USGS DATUM] stream mapping rows:", len(usgs_stream_mapping))

# Load hardcoded NAVD88 offsets (no runtime CSV read) and filter to the mapped stems.
_expected_cols = ["file_stem", "site_no", "index", "offset_ft", "source", "note"]
print("[COMPLETED] [USGS DATUM] hardcoded offset mode active. Source CSV (manual extract):", USGS_OFFSETS_SOURCE_CSV)
usgs_offsets_df = pd.DataFrame(USGS_OFFSETS_HARDCODED_ROWS, columns=_expected_cols)
usgs_offsets_df["file_stem"] = usgs_offsets_df["file_stem"].astype(str).str.strip()
if usgs_stream_mapping is not None and len(usgs_stream_mapping) > 0:
    mapped_stems = set()
    for item in usgs_stream_mapping:
        mapped_stem = str(item.get("file_stem", "")).strip()
        if mapped_stem == "":
            continue
        mapped_stems.add(mapped_stem)
    before_rows = len(usgs_offsets_df)
    usgs_offsets_df = usgs_offsets_df[usgs_offsets_df["file_stem"].isin(mapped_stems)].copy()
    after_rows = len(usgs_offsets_df)
    print("[COMPLETED] [USGS DATUM] filtered hardcoded offsets by mapping rows:", before_rows, "->", after_rows)
print("[COMPLETED] [USGS DATUM] using hardcoded offsets rows:", len(usgs_offsets_df))
usgs_offsets_df = usgs_offsets_df[_expected_cols]

# Build {file_stem: offset_ft}, skipping NaN / non-finite / unrealistic values.
usgs_offset_map = {}
if usgs_offsets_df is None or len(usgs_offsets_df) == 0:
    print("[WARNING] [USGS DATUM] offsets dataframe empty")
else:
    for _, row in usgs_offsets_df.iterrows():
        file_stem = str(row.get("file_stem", "")).strip()
        if file_stem == "":
            continue
        offset_val = pd.to_numeric(row.get("offset_ft", ""), errors="coerce")
        if pd.isna(offset_val):
            continue
        offset_num = float(offset_val)
        if not np.isfinite(offset_num):
            continue
        if abs(offset_num) > USGS_DATUM_OFFSET_ABS_MAX:
            print("[WARNING] [USGS DATUM] skip unrealistic offset in CSV for", file_stem, "offset", offset_num)
            continue
        usgs_offset_map[file_stem] = offset_num
    print("[COMPLETED] [USGS DATUM] usable offsets in map:", len(usgs_offset_map))

# Apply offsets to each USGS stream CSV -> USGS_Cleaned, and copy into Cleaned.
os.makedirs(USGS_CLEANED_DIR, exist_ok=True)
saved_count = 0
copied_count = 0
skipped_count = 0
for item in usgs_stream_mapping:
    file_stem = item["file_stem"]
    site_no = item["site_no"]
    idx = item["index"]
    in_csv = os.path.join(USGS_RAW_DIR, file_stem + "_stream.csv")
    out_csv = os.path.join(USGS_CLEANED_DIR, file_stem + "_stream.csv")
    print("--------------------------------------------------------")
    print("[COMPLETED] [USGS DATUM] apply", file_stem, "site", site_no, "idx", idx)
    if not os.path.isfile(in_csv):
        print("[WARNING] [USGS DATUM] missing input stream file:", in_csv)
        skipped_count = skipped_count + 1
        continue
    try:
        df = pd.read_csv(in_csv)
    except Exception as exc:
        print("[ERROR] [USGS DATUM] read failed:", in_csv, "error", repr(exc))
        skipped_count = skipped_count + 1
        continue
    if df.empty:
        print("[WARNING] [USGS DATUM] empty input file:", in_csv)
        skipped_count = skipped_count + 1
        continue
    df.columns = [str(c).strip() for c in df.columns]
    ts_col = None
    for candidate in ["timestamp_utc", "timestamp_iso", "Timestamp", "timestamp", "time", "datetime", "date"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        print("[ERROR] [USGS DATUM] no timestamp column in:", in_csv)
        skipped_count = skipped_count + 1
        continue
    val_col = None
    for candidate in ["value_raw", "StreamLevel", "stage_ft", "value", "water_ft"]:
        if candidate in df.columns:
            val_col = candidate
            break
    if val_col is None:
        other_cols = [c for c in df.columns if c != ts_col]
        if len(other_cols) == 0:
            print("[ERROR] [USGS DATUM] no value column in:", in_csv)
            skipped_count = skipped_count + 1
            continue
        val_col = other_cols[-1]
    df_one = df[[ts_col, val_col]].copy()
    df_one, idx_utc = parse_timestamp_column_to_utc(df_one, ts_col)
    raw_vals = pd.to_numeric(df_one[val_col], errors="coerce")
    work = pd.DataFrame()
    work["timestamp_utc"] = idx_utc
    work["raw_val_ft"] = raw_vals.values
    work = work.dropna(subset=["timestamp_utc", "raw_val_ft"])
    if work.empty:
        print("[WARNING] [USGS DATUM] no valid rows after parsing:", in_csv)
        skipped_count = skipped_count + 1
        continue
    offset_ft = float(usgs_offset_map.get(file_stem, 0.0))
    if file_stem in usgs_offset_map:
        print("[COMPLETED] [USGS DATUM] using offset_ft", f"{offset_ft:+.3f}", "for", file_stem)
    else:
        print("[WARNING] [USGS DATUM] no offset for", file_stem, "using 0.0")
    work["stage_ft"] = work["raw_val_ft"] + offset_ft
    work = work.sort_values("timestamp_utc")
    work = work.drop_duplicates(subset=["timestamp_utc"], keep="last")
    work["timestamp_local"] = work["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    out = work[["timestamp_utc", "timestamp_local", "stage_ft"]].copy()
    try:
        out.to_csv(out_csv, index=False)
        print("[COMPLETED] [USGS DATUM] wrote corrected:", out_csv, "rows", len(out))
        saved_count = saved_count + 1
    except Exception as exc:
        print("[ERROR] [USGS DATUM] write failed:", out_csv, "error", repr(exc))
        skipped_count = skipped_count + 1
        continue
    dst_csv = os.path.join(CLEAN_DIR, os.path.basename(out_csv))
    try:
        shutil.copy2(out_csv, dst_csv)
        print("[COMPLETED] [USGS DATUM] copied to cleaned:", dst_csv)
        copied_count = copied_count + 1
    except Exception as exc:
        print("[WARNING] [USGS DATUM] copy failed:", out_csv, "->", dst_csv, "error", repr(exc))
print("[COMPLETED] [USGS DATUM] apply summary saved", saved_count, "copied", copied_count, "skipped", skipped_count)

# =============================================================================
# Prepare Untrimmed Postprocessed -> Trimmed directory (compatibility)
# =============================================================================
print("\n[COMPLETED] START Prepare Untrimmed Postprocessed")
print("[UNTRIMMED] Clearing previous files in:", TRIMMED_DIR)
for f in glob.glob(os.path.join(TRIMMED_DIR, "*.csv")):
    os.remove(f)

source_files = sorted(glob.glob(os.path.join(POSTPROCESSED_DIR, "*.csv")))
summary_rows = []

if len(source_files) == 0:
    raise RuntimeError("[UNTRIMMED] No postprocessed files found")

for fp in source_files:
    fname = os.path.basename(fp)

    try:
        df = pd.read_csv(fp)
    except Exception as exc:
        print("[ERROR] [UNTRIMMED] Read failed:", fp, "error", repr(exc))
        continue

    if "timestamp_utc" not in df.columns:
        print("[WARNING] [UNTRIMMED] Missing timestamp_utc, skipping:", fp)
        continue

    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    valid_mask = ts.notna()
    if not valid_mask.any():
        print("[WARNING] [UNTRIMMED] All timestamps invalid:", fp)
        continue

    df = df.loc[valid_mask].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc")
    df = df[~df["timestamp_utc"].duplicated(keep="first")]

    out_path = os.path.join(TRIMMED_DIR, fname)
    df.to_csv(out_path, index=False)
    print("[COMPLETED] [UNTRIMMED] Saved in:", out_path, "rows:", len(df))

    summary_rows.append({
        "file": fname,
        "start_utc": str(df["timestamp_utc"].iloc[0]),
        "end_utc": str(df["timestamp_utc"].iloc[-1]),
        "rows": int(len(df)),
    })

if len(summary_rows) > 0:
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(TRIMMED_DIR, "untrimmed_summary.csv")
    sum_df.to_csv(sum_csv, index=False)
    print("[COMPLETED] [UNTRIMMED] Summary written:", sum_csv)
else:
    raise RuntimeError("[UNTRIMMED] No valid files were prepared")

# =============================================================================
# Trim Postprocessed -> Selected stations (cutoff based, notebook-style)
# =============================================================================
print("\n[COMPLETED] START Trim Postprocessed for selected stations")
print("[SELECTED] Clearing previous files in:", POSTPROCESSED_SELECTED_DIR)
for f in glob.glob(os.path.join(POSTPROCESSED_SELECTED_DIR, "*.csv")):
    os.remove(f)

selected_trim_files = sorted(glob.glob(os.path.join(POSTPROCESSED_DIR, "*.csv")))
selected_summary_rows = []

if len(selected_trim_files) == 0:
    raise RuntimeError("[SELECTED] No postprocessed files found")

for fp in selected_trim_files:
    fname = os.path.basename(fp)

    try:
        df = pd.read_csv(fp)
    except Exception as exc:
        print("[ERROR] [SELECTED] Read failed:", fp, "error", repr(exc))
        continue

    if "timestamp_utc" not in df.columns:
        print("[WARNING] [SELECTED] Missing timestamp_utc:", fp)
        continue

    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    valid_mask = ts.notna()
    if not valid_mask.any():
        print("[WARNING] [SELECTED] All timestamps invalid:", fp)
        continue

    df = df.loc[valid_mask].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc")
    df = df[~df["timestamp_utc"].duplicated(keep="first")]

    orig_start = df["timestamp_utc"].iloc[0]
    orig_end = df["timestamp_utc"].iloc[-1]

    trim_start = SELECTED_CUTOFF_TS
    if trim_start is None:
        trim_start = orig_start
        print("[WARNING] [SELECTED] SELECTED_CUTOFF_TS is None, using file start:", trim_start)
    else:
        allow_start = trim_start + pd.Timedelta(hours=CUTOFF_RELAX_HOURS)
        if orig_start > allow_start:
            print("[WARNING] [SELECTED] Starts after cutoff + grace, skipping:", fp)
            continue

    trim_end = END_TS
    if trim_end is None:
        trim_end = orig_end

    trimmed = df.loc[(df["timestamp_utc"] >= trim_start) & (df["timestamp_utc"] <= trim_end)].copy()
    if trimmed.empty:
        print("[WARNING] [SELECTED] Empty after trim:", fp)
        continue

    out_path = os.path.join(POSTPROCESSED_SELECTED_DIR, fname)
    trimmed.to_csv(out_path, index=False)
    print("[COMPLETED] [SELECTED] Saved:", out_path, "rows:", len(trimmed))

    selected_summary_rows.append({
        "file": fname,
        "original_start_utc": str(orig_start),
        "original_end_utc": str(orig_end),
        "new_start_utc": str(trimmed["timestamp_utc"].iloc[0]),
        "new_end_utc": str(trimmed["timestamp_utc"].iloc[-1]),
        "rows_original": int(len(df)),
        "rows_trimmed": int(len(trimmed)),
        "rows_dropped": int(len(df) - len(trimmed)),
    })

if len(selected_summary_rows) > 0:
    selected_sum_df = pd.DataFrame(selected_summary_rows)
    selected_sum_csv = os.path.join(POSTPROCESSED_SELECTED_DIR, "trim_summary_selected.csv")
    selected_sum_df.to_csv(selected_sum_csv, index=False)
    print("[COMPLETED] [SELECTED] Summary written:", selected_sum_csv)
else:
    raise RuntimeError("[SELECTED] No files passed cutoff trimming")

########## Make Matrix
print("\n[COMPLETED] START Make Matrix")

def build_feature_matrix_from_csvs(
    source_dir,
    log_tag,
    start_message,
    empty_error_message,
    range_error_message,
    series_error_message,
    schema_path,
    engineered_base_path,
    engineered_done_message,
    initial_grid_start,
    grid_floor_minimum,
    start_message_path,
    save_matrix_message_path,
    print_all_done_after_schema,
):
    if start_message_path is None:
        print(start_message)
    else:
        print(start_message, start_message_path)

    source_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    if len(source_files) == 0:
        raise RuntimeError(empty_error_message)

    grid_start = initial_grid_start
    grid_end = None
    features_meta = []

    for p in source_files:
        kind = detect_kind_from_filename(p)
        if kind is None:
            continue
        if kind not in FEATURE_KINDS:
            print(log_tag, "Skipping kind not in FEATURE_KINDS:", kind, "file", p)
            continue

        vcol = CLEAN_COLMAP[kind]
        usecols = ["timestamp_utc", vcol]

        try:
            header_cols = pd.read_csv(p, nrows=0).columns
            df = pd.read_csv(p, usecols=[c for c in usecols if c in header_cols])
        except Exception as exc:
            print("[WARNING]", log_tag, "Read failed:", p, "error", repr(exc))
            continue

        if "timestamp_utc" not in df.columns:
            continue
        if vcol not in df.columns:
            continue

        t = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        t = t.dropna()
        if len(t) == 0:
            continue

        tmin = t.min()
        tmax = t.max()

        if grid_end is None or tmax > grid_end:
            grid_end = tmax

        if grid_start is None:
            grid_start = tmin
        else:
            if tmin < grid_start:
                grid_start = tmin

        if grid_floor_minimum is not None:
            if grid_start < grid_floor_minimum:
                grid_start = grid_floor_minimum

        station = os.path.basename(p).split("_")[0]
        features_meta.append({
            "station": station,
            "kind": kind,
            "path": p,
        })

    if grid_start is None or grid_end is None:
        raise RuntimeError(range_error_message)

    grid_start = pd.Timestamp(grid_start).floor(freq=RESAMPLE_RULE)
    grid_end = pd.Timestamp(grid_end).floor(freq=RESAMPLE_RULE)
    time_index = pd.date_range(start=grid_start, end=grid_end, freq=RESAMPLE_RULE, tz="UTC")

    series_dict = {}
    n_ok = 0

    for meta in features_meta:
        p = meta["path"]
        kind = meta["kind"]
        if kind not in FEATURE_KINDS:
            print(log_tag, "Skipping kind not in FEATURE_KINDS:", kind, "file", p)
            continue

        vcol = CLEAN_COLMAP[kind]

        try:
            df = pd.read_csv(p, usecols=["timestamp_utc", vcol])
        except Exception as exc:
            print("[WARNING]", log_tag, "Read failed:", p, "error", repr(exc))
            continue

        t = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        mask = t.notna()
        df = df.loc[mask]
        t = t.loc[mask]

        s = pd.Series(df[vcol].values, index=t)
        s = s[~s.index.duplicated(keep="first")]
        s = s.sort_index()
        s = s.reindex(time_index)
        s = pd.to_numeric(s, errors="coerce")

        if kind in ["pump", "gate"]:
            s = s.round().clip(lower=0, upper=1)
        elif kind == "rain":
            s = s.clip(lower=0)
            # QC CLIP (deployed 2026-07-07): this alignment path had NO spike clip at all (only the
            # CLEAN path did). Mask physically impossible 15-min totals to NaN here too.
            insane_rain = s > CLEAN_SANITY["rain_in_max_per_15m"]
            if bool(insane_rain.any()):
                print("[WARNING]", log_tag, "Rain spikes masked:", int(insane_rain.sum()),
                      "max", round(float(s.max()), 2), "in ->", meta["station"] + "_" + vcol)
                s = s.mask(insane_rain)
        else:
            s = s.astype("float32")

        s = s.astype("float32")

        colname = meta["station"] + "_" + vcol
        series_dict[colname] = s
        n_ok = n_ok + 1

    print("[COMPLETED]", log_tag, "Aligned usable series:", n_ok)

    if len(series_dict) == 0:
        raise RuntimeError(series_error_message)

    matrix = pd.DataFrame(series_dict, index=time_index)
    nan_ratio = matrix.isna().mean().sort_values(ascending=False)
    print("[COMPLETED]", log_tag, "NaN ratio worst 10:\n", nan_ratio.head(10))

    target_cols = [c for c in matrix.columns if c.endswith("_stage_ft")]

    if save_matrix_message_path is not None:
        print("[COMPLETED]", log_tag, "Saving feature matrix to:", save_matrix_message_path)

    with open(schema_path, "w") as f:
        json.dump({
            "run_stamp": RUN_STAMP,
            "resample_rule": RESAMPLE_RULE,
            "start_utc": str(grid_start),
            "end_utc": str(grid_end),
            "columns": list(matrix.columns),
            "target_cols": target_cols,
        }, f, indent=2)

    print("[COMPLETED]", log_tag, "Feature schema written:", schema_path)

    if print_all_done_after_schema:
        print("[COMPLETED] ALL DONE")

    rain_columns = [col for col in matrix.columns if re.search(r'_rain(_|$)', col)]
    for col in rain_columns:
        matrix[f'{col}_accum_24h'] = matrix[col].rolling(window=24).sum()
        matrix[f'{col}_accum_12h'] = matrix[col].rolling(window=12).sum()
        matrix[f'{col}_accum_6h'] = matrix[col].rolling(window=6).sum()
        matrix[f'{col}_accum_3h'] = matrix[col].rolling(window=3).sum()
        matrix[f'{col}_accum_1h'] = matrix[col].rolling(window=1).sum()

    engineered_out_path = engineered_base_path.replace(".pkl", "_feature_engineered.pkl")
    matrix.to_pickle(engineered_out_path)
    print("[COMPLETED]", log_tag, engineered_done_message, engineered_out_path)

    return matrix


# =============================================================================
# Feature matrix (AI output only, no trimming)
# =============================================================================
M = build_feature_matrix_from_csvs(
    source_dir=TRIMMED_DIR,
    log_tag="[FEATURE]",
    start_message="\n[COMPLETED] START Build feature matrix without trimming",
    empty_error_message="[FEATURE] No untrimmed files found",
    range_error_message="[FEATURE] No valid time range found",
    series_error_message="[FEATURE] No series to build matrix",
    schema_path=os.path.join(MASTER_OUT_DIR, "feature_schema.json"),
    engineered_base_path="./Prepared_Global_Matrix//global_features_all_stations.pkl",
    engineered_done_message="ALL-STATIONS engineered matrix written:",
    initial_grid_start=None,
    grid_floor_minimum=ALLSTN_GRID_FLOOR,
    start_message_path=None,
    save_matrix_message_path=os.path.join(RUN_FEATURE_DIR, "feature_matrix.csv"),
    print_all_done_after_schema=True,
)

# =============================================================================
# Feature matrix (Selected stations only, trimmed with cutoff)
# =============================================================================
M_selected = build_feature_matrix_from_csvs(
    source_dir=POSTPROCESSED_SELECTED_DIR,
    log_tag="[FEATURE-SELECTED]",
    start_message="\n[COMPLETED] START Build selected-stations feature matrix from:",
    empty_error_message="[FEATURE-SELECTED] No selected-station trimmed files found",
    range_error_message="[FEATURE-SELECTED] No valid time range found",
    series_error_message="[FEATURE-SELECTED] No series to build matrix",
    schema_path=os.path.join(MASTER_OUT_DIR, "feature_schema_selected_stations.json"),
    engineered_base_path="./Prepared_Global_Matrix//global_features_selected_stations.pkl",
    engineered_done_message="Engineered matrix written:",
    initial_grid_start=SELECTED_CUTOFF_TS,
    grid_floor_minimum=None,
    start_message_path=POSTPROCESSED_SELECTED_DIR,
    save_matrix_message_path=None,
    print_all_done_after_schema=False,
)
