"""
Phase 1 Runner: Quantify heuristic failure on LongBench prompts.

Produces:
  1. Counterfactual importance scores for N_PROMPTS prompts
  2. H2O and SnapKV heuristic predictions for the same prompts
  3. Comparison metrics (AUC-ROC, top-K recall)
  4. Opening figure: heuristic score vs counterfactual score scatter

For a quick sanity check before the full run, set:
  N_PROMPTS = 3
  SAMPLE_FRACTION = 0.1

Runtime: ~44 GPU hrs for N_PROMPTS=50, SAMPLE_FRACTION=0.5 on A100 80GB
Cost:    ~$66 on RunPod @ $1.50/hr
"""

import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from semantickv.model.loader import load_model
from semantickv.model.hooks import AttentionCapture
from semantickv.ablation.counterfactual import compute_importance_scores
from semantickv.ablation.scoring import evaluate_heuristic


MODEL_PATH = "./models/llama3-8b-instruct"
OUTPUT_DIR = Path("./data/processed/phase1_results")
N_PROMPTS = 50
KEEP_FRACTION = 0.3
MAX_NEW_TOKENS = 50
SAMPLE_FRACTION = 0.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_evaluation_prompts(n: int = 50):
    """
    Load diverse prompts from LongBench.
    Downloads data.zip from HuggingFace Hub and reads JSONL files directly.
    """
    from huggingface_hub import hf_hub_download
    import json, zipfile

    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )

    def _load_jsonl(task_name):
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"data/{task_name}.jsonl") as f:
                return [json.loads(line) for line in f if line.strip()]

    prompts = []

    for item in _load_jsonl("2wikimqa_e")[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nQuestion: " + item["input"],
            "task": "multi_hop_qa",
            "answer": item["answers"][0] if item["answers"] else "",
        })

    for item in _load_jsonl("qasper_e")[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nQuestion: " + item["input"],
            "task": "document_qa",
            "answer": item["answers"][0] if item["answers"] else "",
        })

    for item in _load_jsonl("multi_news_e")[:n // 3]:
        prompts.append({
            "text": item["context"] + "\n\nSummarize the above:",
            "task": "summarization",
            "answer": item["answers"][0] if item["answers"] else "",
        })

    return prompts[:n]


def plot_scatter(results, output_path: Path):
    """
    Figure 1: heuristic rank vs counterfactual rank scatter, colored by task.
    Low correlation here is the paper's opening argument.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    task_colors = {"multi_hop_qa": "#e74c3c", "document_qa": "#3498db", "summarization": "#2ecc71"}

    for ax, method in zip(axes, ["h2o", "snapkv"]):
        for result in results:
            gt = np.array(result["gt_scores"])
            h = np.array(result[f"{method}_scores"])
            valid = ~np.isnan(gt)
            gt = gt[valid]
            h = h[valid]
            color = task_colors.get(result["task"], "gray")
            ax.scatter(
                np.argsort(np.argsort(h)),      # heuristic rank
                np.argsort(np.argsort(gt)),     # counterfactual rank
                alpha=0.05, s=1, c=color,
            )
        ax.set_xlabel(f"{method.upper()} rank")
        ax.set_ylabel("Counterfactual rank")
        ax.set_title(f"{method.upper()} vs Ground Truth\n(perfect = diagonal)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in task_colors.items()]
    axes[1].legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    capture = AttentionCapture(model)

    print(f"Loading {N_PROMPTS} evaluation prompts...")
    prompts = load_evaluation_prompts(N_PROMPTS)

    results = []

    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        print(f"\n--- Prompt {i+1}/{N_PROMPTS} (task: {prompt_data['task']}) ---")

        # Resume: if this prompt was already completed in a prior run, reload and skip.
        result_path = OUTPUT_DIR / f"result_{i:04d}.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
            print(f"  Skipping: already completed (loaded {result_path.name})")
            continue

        tokens = tokenizer(prompt_data["text"], return_tensors="pt")
        seq_len = tokens["input_ids"].shape[1]

        if seq_len < 100 or seq_len > 8000:
            print(f"  Skipping: seq_len={seq_len} out of range")
            continue

        # Step 1: ground truth counterfactual importance
        print(f"  Computing counterfactual importance (seq_len={seq_len})...")
        importance_data = compute_importance_scores(
            model, tokenizer, prompt_data["text"],
            max_new_tokens=MAX_NEW_TOKENS,
            sample_fraction=SAMPLE_FRACTION,
        )

        # Step 2: capture attention statistics (memory-efficient: one layer at a time)
        print("  Capturing attention weights for heuristics...")
        inputs = tokenizer(prompt_data["text"], return_tensors="pt").to(model.device)
        with capture.capture():
            with torch.no_grad():
                _ = model(**inputs, output_attentions=True)

        h2o_scores    = capture.get_h2o_scores()
        snapkv_scores = capture.get_snapkv_scores()

        if h2o_scores is None or snapkv_scores is None:
            print("  Warning: no attention weights captured, skipping.")
            continue

        # Step 3: heuristic scores already computed inside the hook
        gt_scores = importance_data["importance_scores"]

        # Step 4: evaluate
        h2o_eval = evaluate_heuristic(gt_scores, h2o_scores, KEEP_FRACTION, "H2O")
        snapkv_eval = evaluate_heuristic(gt_scores, snapkv_scores, KEEP_FRACTION, "SnapKV")

        result = {
            "prompt_id": i,
            "task": prompt_data["task"],
            "seq_len": seq_len,
            "gt_scores": gt_scores.tolist(),
            "h2o_scores": h2o_scores.tolist(),
            "snapkv_scores": snapkv_scores.tolist(),
            "h2o_eval": h2o_eval,
            "snapkv_eval": snapkv_eval,
            "tokens": importance_data["tokens"],
        }
        results.append(result)

        # Save incrementally — RunPod instances can die
        with open(result_path, "w") as f:
            json.dump(result, f)

        print(f"  H2O    AUC={h2o_eval['auc_roc']:.3f}  Top-K Recall={h2o_eval['top_k_recall']:.3f}")
        print(f"  SnapKV AUC={snapkv_eval['auc_roc']:.3f}  Top-K Recall={snapkv_eval['top_k_recall']:.3f}")

        # Free everything before the next prompt — PyTorch's allocator reservations
        # accumulate across prompts otherwise, OOMing after a few long ones.
        del importance_data, h2o_scores, snapkv_scores, inputs, tokens, gt_scores, result
        capture.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
              f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

    if not results:
        print("No results collected. Check prompt lengths and model loading.")
        return

    # Summary — use nanmean/nanstd because individual prompts can yield NaN AUC
    # when the binarized ground truth is single-class (rare with full sampling).
    print("\n=== PHASE 1 SUMMARY ===")
    for method in ["h2o", "snapkv"]:
        key = f"{method}_eval"
        aucs = np.array([r[key]["auc_roc"] for r in results], dtype=float)
        recalls = np.array([r[key]["top_k_recall"] for r in results], dtype=float)
        n_valid = int(np.sum(~np.isnan(aucs)))
        print(f"{method.upper():8s}  Mean AUC={np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}  "
              f"Mean Top-K Recall={np.nanmean(recalls):.3f}±{np.nanstd(recalls):.3f}  "
              f"(AUC over {n_valid}/{len(results)} prompts)")

    summary = {
        "n_prompts": len(results),
        "keep_fraction": KEEP_FRACTION,
        "h2o_mean_auc": float(np.nanmean([r["h2o_eval"]["auc_roc"] for r in results])),
        "snapkv_mean_auc": float(np.nanmean([r["snapkv_eval"]["auc_roc"] for r in results])),
        "h2o_mean_recall": float(np.nanmean([r["h2o_eval"]["top_k_recall"] for r in results])),
        "snapkv_mean_recall": float(np.nanmean([r["snapkv_eval"]["top_k_recall"] for r in results])),
    }
    with open(OUTPUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")

    # Opening figure
    plot_scatter(results, OUTPUT_DIR / "figure1_heuristic_vs_counterfactual.png")

    print("\n--- GO / NO-GO GATE ---")
    h2o_auc = summary["h2o_mean_auc"]
    snapkv_auc = summary["snapkv_mean_auc"]
    if h2o_auc < 0.75 and snapkv_auc < 0.75:
        print(f"GO: Both heuristics AUC < 0.75 (H2O={h2o_auc:.3f}, SnapKV={snapkv_auc:.3f})")
        print("    The gap is significant. Proceed to Phase 2.")
    else:
        print(f"CAUTION: AUC higher than expected (H2O={h2o_auc:.3f}, SnapKV={snapkv_auc:.3f})")
        print("    Review scatter plots before committing to Phase 2.")


if __name__ == "__main__":
    main()
