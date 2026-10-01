"""Copy the result-figure scripts of Revisions 8, 9 and 13 into Revision 14, pointed at the anchor-fix rerun.

Revision 14 rebuilds every result figure after the anchor fix (Farid, 2026-09-24). The figure recipes are not
rewritten. Each script is copied with two kinds of counted text replacement:
  1. the experiment EXTEND_2026_AUG31_EVENTS_20260918 becomes EXTEND_2026_AUG31_ANCHORFIX_20260924;
  2. the revision folders 20260920_Revision_8, 20260921_Revision_9 and 20260922_Revision_13 become
     20260924_Revision_14, so every output lands in Revision 14 and every read of an earlier revision's work
     folder reads the Revision 14 copy that an earlier script in run_all_figures.sh writes.
Every other revision folder (Revision 6 patterns, the George revision recipes) is left alone.

Script                                  Manuscript figure
source_fig08_hindcast_skill.py                     Figure 8
source_hindcast_examples.py          Figure 9 (Revision 13 legend "hindcast")
source_appendix_figures.py          Figures B1, B2 (file figB3) and C1 (file figC2)
source_simulated_forecast_figures.py       Figure 10, and the work tables Figure 11 reads
source_model_comparison_labels.py          Figure 11 (Revision 13 panel (c) labels)
source_event_only_model_comparison.py    the work tables Figure 12 reads
source_event_only_labels.py          Figure 12 (Revision 13 panel (c) labels)
source_simulated_forecast_examples.py                       Figure 13 (Revision 13, h+24)
source_rainfall_crest_gauge_count.py         Figures 14, 15 and 17
Figures C2 and C3 of the appendix are written by the experiment's figC2_C3_event_hydrographs.py.

Output: one copy per script in this folder, named rev14_<original name>.
"""
import os

PAPER = "project/manuscript/"
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    "20260920_Revision_8/scripts/source_fig08_hindcast_skill.py",
    "20260922_Revision_13/scripts/source_hindcast_examples.py",
    "20260920_Revision_8/scripts/source_appendix_figures.py",
    "20260921_Revision_9/scripts/source_simulated_forecast_figures.py",
    "20260922_Revision_13/scripts/source_model_comparison_labels.py",
    "20260921_Revision_9/scripts/source_event_only_model_comparison.py",
    "20260922_Revision_13/scripts/source_event_only_labels.py",
    "20260922_Revision_13/scripts/source_simulated_forecast_examples.py",
    "20260921_Revision_9/scripts/source_rainfall_crest_gauge_count.py",
]
REPLACEMENTS = [
    ("EXTEND_2026_AUG31_EVENTS_20260918", "EXTEND_2026_AUG31_ANCHORFIX_20260924"),
    ("20260920_Revision_8", "20260924_Revision_14"),
    ("20260921_Revision_9", "20260924_Revision_14"),
    ("20260922_Revision_13", "20260924_Revision_14"),
]

# Edits requested by Farid, applied to one script each and counted: each target must occur exactly once.
# 2026-09-24: Figure 10 title, "replay" becomes "simulated forecast". The second title line is unchanged.
# 2026-09-25: BCSA1299 leaves Figures 9 and 13 because of its stale telemetry (713 masked hours in one stretch,
#   February to March). Only BCSA2400 and BCSU4350 have no masked hours in Bayou Conway, and both are shown already.
#   Farid chose BCSA4419 (175 masked hours, longest gap 95 h). In Figure 9 the recipe's ranking step skips BCSA1299,
#   so its own rule puts BCSA4419, next by mean RMSE and NSE rank, in the same slot. Figure 13's fixed list follows.
FIG09_SKIP_ANCHOR = "COMMON = [\n"
FIG09_SKIP_TUPLE = r'''COMMON = [
    # Revision 14 (Farid, 2026-09-25): BCSA1299 is left out for stale telemetry. The recipe's own ranking then puts
    # BCSA4419 in the fourth Bayou Conway slot.
    ('        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
     '        basin_frame = basin_frame[basin_frame["gauge"] != "BCSA1299"].copy()\n'
     '        basin_frame = basin_frame.sort_values(\n            [\n                "criterion_score",',
     "BCSA1299 left out"),
'''
# 2026-09-28: Figure 10 panel (d) in longitude and latitude. The native recipe's million-feet tick block is replaced by
# fig10_longitude_latitude_ticks.py through one more counted substitution added to the Figure 10 substitution list.
FIG10_OLD_TICKS = (
    "def format_million_feet(value, _position):\n"
    '    return f"{value / 1_000_000.0:.2f}"\n'
    "\n\n"
    "map_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))\n"
    "map_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))\n"
    "map_axis.xaxis.set_major_formatter(FuncFormatter(format_million_feet))\n"
    "map_axis.yaxis.set_major_formatter(FuncFormatter(format_million_feet))\n"
    'map_axis.set_xlabel("Easting (million US ft)", fontsize=12.5)\n'
    'map_axis.set_ylabel("Northing (million US ft)", fontsize=12.5)\n'
)
FIG10_SNIPPET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig10_longitude_latitude_ticks.py")
FIG10_OPEN_ANCHOR = 'spatial_source = open(SPATIAL_RECIPE, encoding="utf-8").read()\n'
FIG10_LIST_ANCHOR = "spatial_source = apply_counted(\n    spatial_source,\n    [\n"
WORDING = {
    "20260921_Revision_9/scripts/source_simulated_forecast_figures.py": [
        ("ST-GNN replay with postprocessing and issue-time HRRR rainfall",
         "ST-GNN simulated forecast with postprocessing and issue-time HRRR rainfall"),
        (FIG10_OPEN_ANCHOR,
         "FIG10_OLD_TICKS = " + repr(FIG10_OLD_TICKS) + "\n"
         + "FIG10_SNIPPET = " + repr(FIG10_SNIPPET) + "\n" + FIG10_OPEN_ANCHOR),
        (FIG10_LIST_ANCHOR,
         FIG10_LIST_ANCHOR
         + "        (FIG10_OLD_TICKS, open(FIG10_SNIPPET, encoding=\"utf-8\").read(), \"latitude and longitude ticks\"),\n"),
        # 2026-09-28: the Mike SSHFS mount was down. The local copy below has the same SHA-256 as the matrix recorded in
        # the anchor-fix archives (f09101a57f3b6d758c187cf7671f239952e82ef26a04df621ec0b83af93e28ce), so the data are
        # identical; only the path changes.
        ('PICKLE = ("project/hpc/Experiments/"\n'
         '          "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"',
         'PICKLE = ("project/Experiments/"\n'
         '          "NIGHTLY_VS_FROZEN_20260911/frozen_inputs/"'),
    ],
    "20260922_Revision_13/scripts/source_hindcast_examples.py": [
        (FIG09_SKIP_ANCHOR, FIG09_SKIP_TUPLE),
    ],
    # 2026-09-25: the event-only figure title, "replay" becomes "simulated operational forecast", as in Figure 11.
    "20260922_Revision_13/scripts/source_event_only_labels.py": [
        ('"Event-only replay, January-August 2026",',
         '"Event-only simulated operational forecast, January-August 2026",'),
    ],
    "20260922_Revision_13/scripts/source_simulated_forecast_examples.py": [
        ('"BCSA1299", "BCSA2400", "BCSU4350", "BCSA4409",',
         '"BCSA4419", "BCSA2400", "BCSU4350", "BCSA4409",'),
    ],
}

for relative in SOURCES:
    source_path = PAPER + relative
    text = open(source_path, encoding="utf-8").read()
    counts = []
    for old, new in REPLACEMENTS:
        counts.append(f"{old}:{text.count(old)}")
        text = text.replace(old, new)
    for old, new in WORDING.get(relative, []):
        found = text.count(old)
        if found != 1:
            raise SystemExit(f"[FATAL] {relative}: expected one '{old}', found {found}")
        text = text.replace(old, new)
        counts.append(f"wording:{found}")
    # replot_fig12 reads no experiment file: its inputs are the work tables of the event comparison script.
    if "EXTEND_2026_AUG31_ANCHORFIX_20260924" not in text and "20260924_Revision_14" not in text:
        raise SystemExit(f"[FATAL] {relative} reads neither the experiment nor Revision 14, check it by hand")
    header = ("# Revision 14 copy made by " + os.path.abspath(__file__) + " on 2026-09-24.\n"
              "# Original: " + source_path + "\n")
    target = os.path.join(HERE, "rev14_" + os.path.basename(relative))
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(header + text)
    print(f"[rev14] {os.path.basename(target)}  " + "  ".join(counts))
