"""Render fig01_workflow verbatim with only the two CPU-chip icons removed.

Chart contract:
- Question: preserve the published workflow while showing GRU and ST-GNN as
  model blocks rather than processor chips.
- Surface: static PNG and PDF.
- Preservation rule: execute the authoritative source after exactly three
  checked substitutions: one output path and two draw calls.
- QA rule: all rendered pixel changes must remain inside the two model icons.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


print("[CONFIG] Current working directory:", os.getcwd(), flush=True)

SOURCE_PATH = (
    "project/"
    "manuscript/"
    "figure_library/recipe_workflow_base.py"
)
OUTPUT_STEM = (
    "project/"
    "manuscript/20260729/"
    "fig01_workflow_no_cpu_verbatim"
)


def draw_model_box_no_cpu(
    axis,
    center_x,
    center_y,
    width,
    height,
    color_value,
    face_value,
):
    """Draw the original chip body without processor pins or inner CPU core."""
    model_box = FancyBboxPatch(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
        ),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0,
        edgecolor=color_value,
        facecolor=face_value,
        zorder=5,
    )
    axis.add_patch(
        model_box,
    )


with open(
    SOURCE_PATH,
    "r",
    encoding="utf-8",
) as source_file:
    authoritative_source = source_file.read()

output_line_original = (
    'OUT = "project/'
    "manuscript/figure_library_outputs/fig01_workflow\""
)
output_line_replacement = (
    'OUT = "project/'
    "manuscript/20260729/"
    "fig01_workflow_no_cpu_verbatim\""
)
gru_call_original = (
    "draw_chip(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, \"#C8DEEF\")"
)
gru_call_replacement = (
    "draw_model_box_no_cpu("
    "ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, \"#C8DEEF\")"
)
stgnn_call_original = (
    "draw_chip(ax, hx[2], hy, 1.2, 1.1, GREEN, \"#C9E8D9\")"
)
stgnn_call_replacement = (
    "draw_model_box_no_cpu("
    "ax, hx[2], hy, 1.2, 1.1, GREEN, \"#C9E8D9\")"
)

replacement_pairs = [
    (
        output_line_original,
        output_line_replacement,
        "output path",
    ),
    (
        gru_call_original,
        gru_call_replacement,
        "GRU icon",
    ),
    (
        stgnn_call_original,
        stgnn_call_replacement,
        "ST-GNN icon",
    ),
]

transformed_source = authoritative_source

for original_text, replacement_text, replacement_name in replacement_pairs:
    occurrence_count = transformed_source.count(
        original_text,
    )

    if occurrence_count != 1:
        raise Exception(
            "Expected exactly one "
            + replacement_name
            + " substitution target, found "
            + str(occurrence_count)
            + "."
        )

    transformed_source = transformed_source.replace(
        original_text,
        replacement_text,
        1,
    )

print(
    "[VERIFY] Applied exactly three source substitutions:",
    ", ".join(
        [
            replacement_name
            for _, _, replacement_name in replacement_pairs
        ]
    ),
    flush=True,
)

execution_globals = {
    "__file__": SOURCE_PATH,
    "__name__": "__main__",
    "draw_model_box_no_cpu": draw_model_box_no_cpu,
}

exec(
    compile(
        transformed_source,
        SOURCE_PATH,
        "exec",
    ),
    execution_globals,
)

expected_outputs = [
    OUTPUT_STEM + ".png",
    OUTPUT_STEM + ".pdf",
]

for expected_output in expected_outputs:
    if not os.path.isfile(expected_output):
        raise Exception(
            "Expected rendered output was not created: "
            + expected_output
        )

    print(
        "[VERIFY] Created:",
        expected_output,
        flush=True,
    )

plt.show()
