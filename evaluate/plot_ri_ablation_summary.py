import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 220,
    "savefig.dpi": 400,
})

CSV_PATH = "./results/ri_physics_ablation/ri_physics_ablation_summary.csv"
SAVE_FIG = "./results/ri_physics_ablation/Figure_RI_predictor_ablation.png"

NAME_MAP = {
    "thermodynamic_support": "Thermodynamic\nsupport",
    "low_level_dynamics": "Low-level\ndynamics",
    "midlevel_structure": "Midlevel\nstructure",
    "upper_level_outflow": "Upper-level\noutflow",
    "intensity_history": "Intensity\nhistory",
    "track_history": "Track\nhistory",
}

ORDER = [
    "thermodynamic_support",
    "low_level_dynamics",
    "midlevel_structure",
    "upper_level_outflow",
    "intensity_history",
    "track_history",
]


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["Obs_RI"] == 1].copy()
    df["GroupLabel"] = df["Group"].map(NAME_MAP)
    df["Group"] = pd.Categorical(df["Group"], categories=ORDER, ordered=True)
    df = df.sort_values("Group")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(df["GroupLabel"], df["Drop_RI_Prob"], color="#4C78A8")
    ax.set_ylabel("Mean decrease in RI probability", fontsize=18)
    ax.set_title("RI sensitivity to physics-guided predictor groups", fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(SAVE_FIG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {SAVE_FIG}")


if __name__ == "__main__":
    main()
