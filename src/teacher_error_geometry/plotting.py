"""Publication-candidate static plots, solely from saved summaries."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import read_csv, read_json, sha256, write_json


def figures(summary, output):
    summary, output = Path(summary), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    overview = read_json(summary / "overview.json")
    config = overview["config"]
    rows = list(read_csv(summary / "cells.csv"))
    numeric = ("rho", "n", "mean", "mc_se", "theory", "c", "eta", "a", "b", "theta")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key]) if row[key] else np.nan
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "teacher-error-geometry-p1", "figure.dpi": 120})
    paths = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Date": None} if extension == "svg" else {})
            paths.append(path)
        plt.close(fig)

    def select(block, metric="truth_risk", **kwargs):
        return [r for r in rows if r["block"] == block and r["metric"] == metric and
                all(r[k] == v for k, v in kwargs.items())]

    # 1. Five actual rho panels; curves and MC intervals are distinct artists.
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    for ax, rho in zip(axes, config["A"]["rho"]):
        grid = np.geomspace(10, 1000, 300)
        for kind, color, label in (("axis_visible", "#b54822", "visible"), ("axis_hidden", "#2166ac", "hidden")):
            data = sorted(select("A", rho=rho, teacher=kind), key=lambda r: r["n"])
            expected = np.full_like(grid, config["c"] ** 2) if kind == "axis_visible" else config["c"] ** 2 * (rho * rho + (1 - rho * rho) / (grid - 2))
            ax.plot(grid, expected, color=color, label=f"{label}: theory")
            ax.errorbar([r["n"] for r in data], [r["mean"] for r in data],
                        yerr=[1.96*r["mc_se"] for r in data], fmt="o", ms=4, capsize=3, color=color,
                        label=f"{label}: MC +/-95%")
        ax.set(xscale="log", xlabel="training n", title=rf"$\rho={rho:g}$")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("population truth risk")
    axes[-1].legend(fontsize=8, loc="center right")
    fig.suptitle("A: visible bias persists; hidden errors contribute bias and sampling variance (50 cells)", y=1.05)
    save(fig, "01_truth_by_n")

    # 2. B is paired with A; rho=0 is one reused target, not replication.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    grid = np.linspace(-.9, .9, 300)
    for ax, n in zip(axes, config["B"]["n"]):
        c2 = config["c"] ** 2
        ax.plot(grid, c2*(grid**2 + (1-grid**2)/(n-2)), color="#2166ac", label="hidden theory")
        ax.plot(grid, c2*grid**2, "--", color="#777777", label="hidden infinite-n floor")
        ax.axhline(c2/(n-2), color="#238b45", label="projection-zero theory")
        ax.axhline(0, color="#238b45", ls=":", label="projection-zero infinite-n floor")
        for block, kind, color, marker in (("A", "axis_hidden", "#2166ac", "o"), ("B", "projection_zero", "#238b45", "s")):
            data = sorted(select(block, n=n, teacher=kind), key=lambda r: r["rho"])
            ax.errorbar([r["rho"] for r in data], [r["mean"] for r in data],
                        yerr=[1.96*r["mc_se"] for r in data], fmt=marker, color=color, capsize=3,
                        label=f"{kind}: MC +/-95%")
        ax.set(xlabel=r"$\rho$", title=f"n={n}; 5 hidden + 5 paired projection-zero cells")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("population truth risk")
    axes[1].legend(fontsize=8)
    fig.suptitle("B reuses A data; at rho=0 the hidden and projection-zero teachers coincide", y=1.04)
    save(fig, "02_hidden_projection_by_rho")

    # 3. Easily read concrete rho=0 counterexample with both fitted risks.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for ax, metric, title in zip(axes, ("truth_risk", "teacher_risk"), ("Truth risk", "Teacher agreement MSE")):
        for kind, color in (("axis_visible", "#b54822"), ("axis_hidden", "#2166ac")):
            data = sorted(select("A", metric, rho=0.0, teacher=kind), key=lambda r:r["n"])
            grid = np.geomspace(10, 1000, 300)
            if kind == "axis_visible":
                theory = np.full_like(grid, config["c"]**2 if metric=="truth_risk" else 0)
            else:
                theory = config["c"]**2/(grid-2) + (config["c"]**2 if metric=="teacher_risk" else 0)
            ax.plot(grid, theory, color=color, label=f"{kind}: theory")
            ax.errorbar([r["n"] for r in data], [r["mean"] for r in data],
                        yerr=[1.96*r["mc_se"] for r in data], fmt="o", color=color, capsize=3,
                        label=f"{kind}: MC +/-95%")
        ax.set(xscale="log", xlabel="training n", title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("population MSE")
    fig.suptitle("A, rho=0: matching the teacher can coexist with persistent truth error", y=1.04)
    save(fig, "03_teacher_vs_truth")

    # 4. Exact 3 rho x 4 theta x 2 n = 24 C cells.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout=None)
    for ni, n in enumerate(config["C"]["n"]):
        for column, key in enumerate(("theory", "mean")):
            matrix = np.array([[next(r[key] for r in select("C", n=n, rho=rho) if r["theta"] == theta)
                                for theta in config["C"]["theta"]] for rho in config["C"]["rho"]])
            ax = axes[ni, column]
            plot = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=config["c"]**2, aspect="auto")
            ax.set_xticks(range(4), ["0", "pi/4", "pi/2", "3pi/4"])
            ax.set_yticks(range(3), [f"{r:g}" for r in config["C"]["rho"]])
            ax.set(xlabel="Euclidean direction theta", ylabel="rho",
                   title=f"n={n}; {'theory' if key=='theory' else 'MC mean'}; 12 cells")
            for i in range(3):
                for j in range(4):
                    ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", color="white" if matrix[i,j]<.13 else "black")
            fig.colorbar(plot, ax=ax, label="truth risk", shrink=.85)
    fig.suptitle("C: 24 normalized-angle cells; delta^T Sigma delta=0.25; Euclidean ||delta|| varies", y=1.02)
    save(fig, "04_angle_heatmap")

    # 5. Same normalized eta relation; colors encode actual covariance.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    color_map = {-.75:"#2166ac", 0.0:"#444444", .75:"#b54822"}
    for ax, n in zip(axes, config["C"]["n"]):
        eta = np.linspace(0, 1, 200)
        ax.plot(eta, eta+(1-eta)/(n-2), color="black", lw=1.3, label="eta relation (theory)")
        for ri, rho in enumerate(config["C"]["rho"]):
            data = select("C", n=n, rho=rho)
            # Small marker sizes preserve exact eta positions without jitter.
            ax.errorbar([r["eta"] for r in data], [r["mean"]/config["c"]**2 for r in data],
                        yerr=[1.96*r["mc_se"]/config["c"]**2 for r in data], fmt=["o","s","^"][ri],
                        ms=5, mfc="none", capsize=3, color=color_map[rho], label=f"rho={rho:g}, MC +/-95%")
        ax.set(xlabel="eta = a^2 / c^2", title=f"n={n}; 12 C cells")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("expected truth risk / c^2 (MC estimate)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fixed population teacher error aligns by projected error fraction eta", y=1.04)
    save(fig, "05_eta_collapse")
    result = {"figure_count": 5, "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths],
              "data_source": "cells.csv plus frozen config in overview.json; no evaluation data generated",
              "intervals": "pointwise normal 95%, not simultaneous"}
    write_json(output / "manifest.json", result)
    return result
