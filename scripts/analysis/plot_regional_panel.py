"""
Plotting functions for regional panel analysis (Gini trends, etc).
"""
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def plot_gini(gini_by_year, output_path, show_plot=True):
    """
    Plot the yearly Gini series and optionally save the figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(gini_by_year["year"], gini_by_year["gini"], marker="o", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Gini Coefficient")
    plt.title("Regional NTL Inequality (Gini) Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Gini plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# --- New plotting functions ---
def plot_ntl_histogram(df, output_path, show_plot=True):
    """
    Plot histogram and density of NTL values (all years or a selected year).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    df["NTL"].plot(kind="hist", bins=30, alpha=0.6, density=True, label="Histogram")
    df["NTL"].plot(kind="kde", label="Density", color="red")
    plt.xlabel("NTL Value")
    plt.ylabel("Density")
    plt.title("Distribution of Nighttime Lights (NTL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved NTL histogram to: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_leader_vs_nonleader_trends(df, output_path, show_plot=True):
    """
    Plot time trends of mean NTL for leader vs. non-leader regions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means = df.groupby(["year", "Leader_region"])['NTL'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    for key, group in means.groupby("Leader_region"):
        label = "Leader" if key == 1 else "Non-Leader"
        plt.plot(group["year"], group["NTL"], marker="o", label=label)
    plt.xlabel("Year")
    plt.ylabel("Mean NTL")
    plt.title("NTL Trends: Leader vs. Non-Leader Regions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved leader vs. non-leader trend plot to: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_ntl_boxplot_by_leader(df, output_path, show_plot=True):
    """
    Boxplot of NTL by leader status (all years or a selected year).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    df["Leader_region"] = df["Leader_region"].map({0: "Non-Leader", 1: "Leader"})
    df.boxplot(column="NTL", by="Leader_region", grid=False)
    plt.title("NTL by Leader Status")
    plt.suptitle("")
    plt.xlabel("Region Type")
    plt.ylabel("NTL Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved NTL boxplot by leader status to: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
