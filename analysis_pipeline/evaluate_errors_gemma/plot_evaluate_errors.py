import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Load the data ---
df = pd.read_excel("../error_analysis_data.xlsx", sheet_name="Sheet1")
df = df[df["id"] >= 30]

# Relevant columns
error_col = "Error_type_LLM"       # Column M
example_col = "Example (Error_type + case)"  # Column N

# =============================================
# 1) Horizontal lollipop chart: Count of each Error_type_LLM
# =============================================
df[error_col] = df[error_col].replace("unclear", "indeterminate")
counts = df[error_col].value_counts()
total = counts.sum()

# Sort ascending so largest is on top after inversion
counts_sorted = counts.sort_values(ascending=True)
labels = counts_sorted.index.tolist()
values = counts_sorted.values

# Color palette – distinct, colorblind-friendly
palette = {
    "Size incorrectly identified":          "#E63946",
    "Re-resection classified as new polyp": "#457B9D",
    "Polyp not detected":                   "#2A9D8F",
    "Wrong location identified":            "#E9C46A",
    "indeterminate":                        "#A8DADC",
    "Re-resection classified as en-bloc":   "#264653",
}
colors = [palette.get(l, "#888888") for l in labels]

# --- Figure ---
fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)

y_pos = np.arange(len(labels))

# Horizontal lines (stems)
ax.hlines(y=y_pos, xmin=0, xmax=values, color=colors, linewidth=2.5, zorder=2)

# Dots at the end
ax.scatter(values, y_pos, color=colors, s=120, zorder=3, edgecolors="white", linewidths=1.2)

# Count + percentage labels
for i, v in enumerate(values):
    pct = v / total * 100
    ax.text(v + 0.5, i, f"{v}  ({pct:.0f}%)",
            va="center", fontsize=12, fontweight="600", color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlabel("Number of errors", fontsize=14, labelpad=8)
ax.set_title("Gemma 3 27B errors — human annotation correct", fontsize=15,
             fontweight="bold", pad=28, loc="left")

# Subtitle with total
ax.text(0, 1.02, f"n = {total} errors total",
        transform=ax.transAxes, fontsize=12, color="#666666")

# Styling
ax.set_xlim(0, max(values) + 5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("error_type_counts.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig("error_type_counts.svg", bbox_inches="tight", facecolor="white")
plt.show()

# =============================================
# 2) Table: Unique examples for each error type
# =============================================
# Column N has one canonical example per error type;
# drop duplicates to get a clean lookup table.
examples_df = (
    df[[error_col, example_col]]
    .dropna(subset=[example_col])
    .drop_duplicates(subset=[example_col])
    .sort_values(error_col)
    .reset_index(drop=True)
)

# Pretty-print in the console
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
print("\n===  Error Types with Example Descriptions  ===\n")
print(examples_df.to_string(index=False))

