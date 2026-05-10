"""
All plots for the dense vs sparse vs hybrid RAG comparison.

Always-generated (7 core plots):
  1.  accuracy_comparison_{dataset}.png     — EM/F1/RHR bars with CI per dataset
  2.  latency_comparison.png                — stacked latency bars (all datasets)
  3.  accuracy_latency_tradeoff.png         — F1 vs latency scatter
  4.  f1_distribution_{dataset}.png         — violin+strip per dataset
  5.  failure_mode_breakdown_{dataset}.png  — retrieval/gen error stacks
  6.  pipeline_disagreement_{dataset}.png   — dense vs sparse F1 scatter

Cross-dataset plots (generated when >1 dataset):
  7.  cross_dataset_f1.png          — grouped bar: F1 per pipeline × dataset
  8.  cross_dataset_rhr.png         — grouped bar: RHR per pipeline × dataset
  9.  dataset_delta.png             — bar: TriviaQA F1 minus SQuAD F1 per pipeline

Conditional (ablation flags):
  10. topk_ablation.png
  11. chunk_ablation.png
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PALETTE = {"dense": "#5C6BC0", "sparse": "#EF7C43", "hybrid": "#26A69A"}
FALLBACK = ["#5C6BC0", "#EF7C43", "#26A69A", "#AB47BC", "#EC407A"]
DATASET_MARKERS = {"squad": "o", "trivia_qa": "s"}
DATASET_LABELS  = {"squad": "SQuAD", "trivia_qa": "TriviaQA"}


def _pc(name, i=0):
    return PALETTE.get(name, FALLBACK[i % len(FALLBACK)])


def _save(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_results(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: str,
    ablation_topk_df: Optional[pd.DataFrame] = None,
    ablation_chunk_df: Optional[pd.DataFrame] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    datasets  = list(summary_df["dataset"].unique())
    pipelines = sorted(summary_df["pipeline"].unique())

    # ── Per-dataset plots ─────────────────────────────────────────────────
    for dataset in datasets:
        ds_label = DATASET_LABELS.get(dataset, dataset)
        s_df = summary_df[summary_df["dataset"] == dataset].reset_index(drop=True)
        r_df = results_df[results_df["dataset"] == dataset].reset_index(drop=True)
        pipes = list(s_df["pipeline"])
        colors = [_pc(p, i) for i, p in enumerate(pipes)]
        x = np.arange(len(pipes))

        # 1. Accuracy comparison with CI error bars
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.25
        metrics = [
            ("em",                 "em_ci_low",  "em_ci_high",  "Exact Match"),
            ("f1",                 "f1_ci_low",  "f1_ci_high",  "F1"),
            ("retrieval_hit_rate", "rhr_ci_low", "rhr_ci_high", "Retrieval Hit Rate"),
        ]
        for (col, lo, hi, lbl), offset, hatch in zip(
            metrics, [-width, 0, width], ["", "///", "..."]
        ):
            vals = s_df[col].values
            yerr = [vals - s_df[lo].values, s_df[hi].values - vals]
            ax.bar(x + offset, vals, width=width, label=lbl, hatch=hatch,
                   color=colors, alpha=0.82,
                   yerr=yerr, error_kw=dict(ecolor="black", capsize=4, elinewidth=1.2))
            for xi, v in zip(x, vals):
                ax.text(xi + offset, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(pipes)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
        ax.set_title(f"[{ds_label}] Accuracy Comparison with 95% Bootstrap CI")
        ax.legend(loc="upper right", framealpha=0.9)
        _save(fig, os.path.join(output_dir, f"accuracy_comparison_{dataset}.png"))

        # 2. Per-query F1 violin + strip
        fig, ax = plt.subplots(figsize=(9, 5))
        f1_data = [r_df[r_df["pipeline"] == p]["f1"].values for p in pipes]
        vp = ax.violinplot(f1_data, positions=range(len(pipes)),
                           showmedians=True, showextrema=True)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.55)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
        for part in ("cbars","cmins","cmaxes"):
            vp[part].set_color("black"); vp[part].set_linewidth(1)
        rng = np.random.default_rng(42)
        for i, (data, c) in enumerate(zip(f1_data, colors)):
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(np.full(len(data), i) + jitter, data,
                       alpha=0.35, s=18, color=c, zorder=4)
            ax.scatter(i, np.mean(data), marker="D", s=60, color=c,
                       edgecolors="black", linewidths=0.8, zorder=6,
                       label=f"{pipes[i]} mean={np.mean(data):.2f}")
        ax.set_xticks(range(len(pipes))); ax.set_xticklabels(pipes)
        ax.set_ylabel("F1 Score"); ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"[{ds_label}] Per-Query F1 Distribution\n"
                     "(violin=density, dots=queries, ◆=mean)")
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, os.path.join(output_dir, f"f1_distribution_{dataset}.png"))

        # 3. Failure mode breakdown
        fig, ax = plt.subplots(figsize=(9, 5))
        added = set()
        for i, pipe in enumerate(pipes):
            sub = r_df[r_df["pipeline"] == pipe]
            n = len(sub)
            ok     = ((sub["retrieval_hit"]==1)&(sub["em"]==1)).sum()/n
            gen_f  = ((sub["retrieval_hit"]==1)&(sub["em"]==0)).sum()/n
            ret_f  = (sub["retrieval_hit"]==0).sum()/n
            def lbl(k, t): return t if k not in added else ""
            ax.bar(i, ok,    color="#4caf50", label=lbl("ok",   "Retrieval ✓  Generation ✓"))
            added.add("ok")
            ax.bar(i, gen_f, bottom=ok,       color="#ff9800", label=lbl("gf","Retrieval ✓  Generation ✗"))
            added.add("gf")
            ax.bar(i, ret_f, bottom=ok+gen_f, color="#f44336", label=lbl("rf","Retrieval ✗"))
            added.add("rf")
            for bot, h, c in [(0,ok,"white"),(ok,gen_f,"white"),(ok+gen_f,ret_f,"white")]:
                if h > 0.04:
                    ax.text(i, bot+h/2, f"{h:.0%}", ha="center", va="center",
                            fontsize=9, color=c, fontweight="bold")
        ax.set_xticks(range(len(pipes))); ax.set_xticklabels(pipes)
        ax.set_ylabel("Fraction of queries"); ax.set_ylim(0, 1)
        ax.set_title(f"[{ds_label}] Failure Mode Breakdown per Pipeline")
        ax.legend(loc="lower right", framealpha=0.9)
        _save(fig, os.path.join(output_dir, f"failure_mode_breakdown_{dataset}.png"))

        # 4. Pipeline disagreement scatter (dense vs sparse)
        dense_df  = r_df[r_df["pipeline"]=="dense"].reset_index(drop=True)
        sparse_df = r_df[r_df["pipeline"]=="sparse"].reset_index(drop=True)
        if len(dense_df) > 0 and len(sparse_df) > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(dense_df["f1"], sparse_df["f1"],
                       alpha=0.4, s=25, color="#5C6BC0", edgecolors="none")
            ax.plot([0,1],[0,1],"k--",alpha=0.3,linewidth=1)
            ax.set_xlabel("Dense F1"); ax.set_ylabel("Sparse F1")
            ax.set_title(f"[{ds_label}] Per-Query Agreement: Dense vs Sparse F1")
            ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
            n = len(dense_df)
            dw = (dense_df["f1"]>sparse_df["f1"]).sum()
            sw = (sparse_df["f1"]>dense_df["f1"]).sum()
            ties = (dense_df["f1"]==sparse_df["f1"]).sum()
            bf   = ((dense_df["f1"]==0)&(sparse_df["f1"]==0)).sum()
            kw = dict(transform=ax.transAxes, fontsize=9)
            ax.text(0.97,0.03,f"Dense wins: {dw}/{n}",ha="right",**kw)
            ax.text(0.03,0.97,f"Sparse wins: {sw}/{n}",ha="left",**kw)
            ax.text(0.03,0.03,f"Ties: {ties}\nBoth fail: {bf}",ha="left",va="bottom",**kw)
            _save(fig, os.path.join(output_dir, f"pipeline_disagreement_{dataset}.png"))

    # ── Shared latency plot (one bar per pipeline, averaged across datasets) ──
    lat_agg = (summary_df.groupby("pipeline", as_index=False)
               .agg(avg_retrieval_latency_sec=("avg_retrieval_latency_sec","mean"),
                    avg_generation_latency_sec=("avg_generation_latency_sec","mean")))
    fig, ax = plt.subplots(figsize=(8,5))
    pipes_l = list(lat_agg["pipeline"])
    colors_l = [_pc(p,i) for i,p in enumerate(pipes_l)]
    r = lat_agg["avg_retrieval_latency_sec"].values
    g = lat_agg["avg_generation_latency_sec"].values
    ax.bar(pipes_l, r, label="Retrieval", color=colors_l, alpha=0.7)
    ax.bar(pipes_l, g, bottom=r, label="Generation", color=colors_l, alpha=1.0, hatch="///")
    for i,(rv,gv) in enumerate(zip(r,g)):
        ax.text(i,rv+gv+0.003,f"{rv+gv:.3f}s",ha="center",va="bottom",fontsize=9)
    ax.set_ylabel("Seconds per query")
    ax.set_title("Latency Breakdown: Retrieval vs Generation\n(averaged across all datasets)")
    ax.legend()
    _save(fig, os.path.join(output_dir, "latency_comparison.png"))

    # ── Accuracy vs Latency tradeoff ──────────────────────────────────────
    lat_f1 = (summary_df.groupby("pipeline", as_index=False)
              .agg(f1=("f1","mean"), lat=("avg_total_latency_sec","mean")))
    fig, ax = plt.subplots(figsize=(8,5))
    for i,(_, row) in enumerate(lat_f1.iterrows()):
        c = _pc(row["pipeline"], i)
        ax.scatter(row["lat"], row["f1"], s=160, color=c, zorder=5)
        ax.annotate(row["pipeline"], (row["lat"], row["f1"]),
                    textcoords="offset points", xytext=(8,6), fontsize=10)
    ax.set_xlabel("Average Total Latency (sec)")
    ax.set_ylabel("Mean F1 Score (averaged across datasets)")
    ax.set_title("Accuracy vs Latency Trade-off")
    _save(fig, os.path.join(output_dir, "accuracy_latency_tradeoff.png"))

    # ── Cross-dataset plots (only when >1 dataset) ────────────────────────
    if len(datasets) > 1:
        _plot_cross_dataset(summary_df, datasets, pipelines, output_dir)

    # ── Ablation plots ────────────────────────────────────────────────────
    if ablation_topk_df is not None and not ablation_topk_df.empty:
        _plot_topk_ablation(ablation_topk_df, output_dir)

    if ablation_chunk_df is not None and not ablation_chunk_df.empty:
        _plot_chunk_ablation(ablation_chunk_df, output_dir)


def _plot_cross_dataset(summary_df, datasets, pipelines, output_dir):
    """
    Three cross-dataset plots:
      A) Grouped bar: F1 per pipeline, grouped by dataset
      B) Grouped bar: RHR per pipeline, grouped by dataset
      C) Delta bar: TriviaQA F1 − SQuAD F1 per pipeline
         (positive = dense generalises; negative = BM25 advantage collapses)
    """
    ds_labels = [DATASET_LABELS.get(d, d) for d in datasets]

    ci_prefix = {
        "f1": "f1",
        "retrieval_hit_rate": "rhr",
    }

    for metric, ylabel, fname in [
        ("f1",                 "Mean F1 Score",         "cross_dataset_f1.png"),
        ("retrieval_hit_rate", "Retrieval Hit Rate",     "cross_dataset_rhr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        n_ds   = len(datasets)
        n_pipe = len(pipelines)
        total_w = 0.7
        w = total_w / n_ds
        x = np.arange(n_pipe)

        for di, (ds, ds_lbl) in enumerate(zip(datasets, ds_labels)):
            sub = summary_df[summary_df["dataset"]==ds].set_index("pipeline")
            prefix = ci_prefix.get(metric, metric)
            vals = [sub.loc[p, metric] if p in sub.index else 0 for p in pipelines]
            lo   = [sub.loc[p, f"{prefix}_ci_low"]  if p in sub.index else 0 for p in pipelines]
            hi   = [sub.loc[p, f"{prefix}_ci_high"] if p in sub.index else 0 for p in pipelines]
            yerr = [np.array(vals)-np.array(lo), np.array(hi)-np.array(vals)]
            offset = (di - (n_ds-1)/2) * w
            bars = ax.bar(x + offset, vals, width=w*0.85,
                          label=ds_lbl,
                          color=[_pc(p, i) for i, p in enumerate(pipelines)],
                          alpha=0.75 + 0.25*di,
                          yerr=yerr,
                          error_kw=dict(ecolor="black", capsize=3, elinewidth=1))
            for xi, v in zip(x, vals):
                ax.text(xi + offset, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x); ax.set_xticklabels(pipelines)
        ax.set_ylabel(ylabel); ax.set_ylim(0, 1.15)
        ax.set_title(f"Cross-Dataset Comparison: {ylabel}\n"
                     f"(each pipeline shown for each dataset)")
        ax.legend(title="Dataset", framealpha=0.9)

        # Add CI legend note
        ax.text(0.98, 0.01, "Error bars = 95% bootstrap CI",
                transform=ax.transAxes, ha="right", fontsize=7, style="italic")
        _save(fig, os.path.join(output_dir, fname))

    # ── Delta bar: TriviaQA F1 minus SQuAD F1 ────────────────────────────
    if "squad" in datasets and "trivia_qa" in datasets:
        sq  = summary_df[summary_df["dataset"]=="squad"].set_index("pipeline")
        tqa = summary_df[summary_df["dataset"]=="trivia_qa"].set_index("pipeline")
        common = [p for p in pipelines if p in sq.index and p in tqa.index]

        deltas = [tqa.loc[p,"f1"] - sq.loc[p,"f1"] for p in common]
        colors_d = ["#4caf50" if d >= 0 else "#f44336" for d in deltas]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(common, deltas, color=colors_d, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x()+bar.get_width()/2,
                    d + (0.005 if d >= 0 else -0.015),
                    f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("F1 Delta (TriviaQA − SQuAD)")
        ax.set_title("Generalisation Gap: TriviaQA F1 − SQuAD F1 per Pipeline\n"
                     "(positive = pipeline performs better on TriviaQA; "
                     "negative = drops on TriviaQA)")
        _save(fig, os.path.join(output_dir, "dataset_delta.png"))


def _plot_topk_ablation(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, title, ax in zip(
        ["f1", "retrieval_hit_rate"],
        ["Mean F1 vs top_k", "Retrieval Hit Rate vs top_k"], axes
    ):
        for i, pipe in enumerate(df["pipeline"].unique()):
            sub = df[df["pipeline"]==pipe].sort_values("top_k")
            # Handle per-dataset if column exists
            if "dataset" in df.columns:
                for j, ds in enumerate(df["dataset"].unique()):
                    s = sub[sub["dataset"]==ds].sort_values("top_k")
                    lbl = DATASET_LABELS.get(ds, ds)
                    ax.plot(s["top_k"], s[metric], marker="o",
                            color=_pc(pipe,i),
                            linestyle="-" if j==0 else "--",
                            label=f"{pipe} ({lbl})", linewidth=2)
            else:
                ax.plot(sub["top_k"], sub[metric], marker="o",
                        color=_pc(pipe,i), label=pipe, linewidth=2)
        ax.set_xlabel("top_k"); ax.set_ylabel(metric)
        ax.set_title(title); ax.legend(fontsize=7)
        ax.set_xticks(sorted(df["top_k"].unique()))
    fig.suptitle("Top-k Ablation", fontsize=12)
    _save(fig, os.path.join(output_dir, "topk_ablation.png"))


def _plot_chunk_ablation(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric, title, ax in zip(
        ["f1", "retrieval_hit_rate"],
        ["Mean F1 vs chunk size", "RHR vs chunk size"], axes
    ):
        for i, pipe in enumerate(df["pipeline"].unique()):
            sub = df[df["pipeline"]==pipe].sort_values("chunk_size")
            ax.plot(sub["chunk_size"], sub[metric], marker="s",
                    color=_pc(pipe,i), label=pipe, linewidth=2)
        ax.set_xlabel("Chunk size (tokens)"); ax.set_ylabel(metric)
        ax.set_title(title); ax.legend()
        ax.set_xticks(sorted(df["chunk_size"].unique()))
    fig.suptitle("Chunk-Size Ablation", fontsize=12)
    _save(fig, os.path.join(output_dir, "chunk_ablation.png"))
