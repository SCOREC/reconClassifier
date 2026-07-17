"""
Optuna hyperparameter tuner for XPointMLTest.py

Usage:
    python optuna_tuner.py \
        --paramFile /path/to/params.txt \
        --xptCacheDir /path/to/cache \
        --n-trials 50 \
        --study-name xpoint-tuning \
        --db sqlite:///optuna_xpoint.db

The script wraps the existing training pipeline and searches over:
    - Learning rate
    - Weight decay
    - Dropout rate
    - Batch size
    - Patch size
    - Base channels (model capacity)
    - Scheduler type + params
    - Augmentation positive-patch ratio

Results persist in a SQLite database so you can:
    - Resume after SSH disconnects
    - Analyze results with Optuna's built-in visualization
    - Run multiple workers in parallel (each with their own process)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("Optuna not installed. Install with:")
    print("  pip install optuna --break-system-packages")
    print("Optional visualization: pip install plotly --break-system-packages")
    sys.exit(1)

# Import from existing codebase
from XPointMLTest import (
    XPointDataset,
    XPointPatchDataset,
    UNet,
    DiceLoss,
    train_one_epoch,
    validate_one_epoch,
    set_seed,
)
from eval_metrics import evaluate_model_on_dataset
from ci_tests import SyntheticXPointDataset


def objective(trial, args):
    """
    Optuna objective function. Trains the model with trial-suggested
    hyperparameters and returns the best validation loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Seed: vary per trial for diversity, but keep reproducible ---
    seed = args.seed
    if seed is not None:
        set_seed(seed + trial.number)

    # 1. Suggest ALL hyperparameters in one place

    # Model architecture
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)

    # Optimizer
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Data pipeline
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    patch_size = trial.suggest_categorical("patch_size", [48, 64, 96])
    pos_ratio = trial.suggest_float("pos_ratio", 0.3, 0.7)

    # Scheduler
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "plateau"])

    # 2. Load data (pre-loaded and cached on args to avoid repeated I/O)
    
    if args.smoke_test:
        train_dataset = SyntheticXPointDataset(nframes=10, shape=(64, 64), nxpoints=3)
        val_dataset = SyntheticXPointDataset(
            nframes=2, shape=(64, 64), nxpoints=3, seed=123
        )
    else:
        train_dataset = args._train_dataset
        val_dataset = args._val_dataset

    train_crop = XPointPatchDataset(
        train_dataset,
        patch=patch_size,
        pos_ratio=pos_ratio,
        retries=30,
        augment=True,
        seed=seed,
    )
    val_crop = XPointPatchDataset(
        val_dataset,
        patch=patch_size,
        pos_ratio=0.5,
        retries=30,
        augment=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_crop, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_crop, batch_size=batch_size, shuffle=False, num_workers=0
    )

   
    # 3. Create model, optimizer, scheduler

    model = UNet(
        input_channels=4, base_channels=base_channels, dropout_rate=dropout_rate
    ).to(device)

    num_epochs = args.epochs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "plateau":
        plateau_factor = trial.suggest_float("plateau_factor", 0.2, 0.8)
        plateau_patience = trial.suggest_int("plateau_patience", 3, 15)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=1e-6,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

    criterion = DiceLoss(smooth=1.0)

    # --- AMP setup ---
    use_amp = args.use_amp and torch.cuda.is_available()
    amp_dtype = (
        torch.bfloat16
        if args.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))


    # 4. Training loop with Optuna pruning

    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience
    epochs_trained = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            amp_dtype,
        )

        # Reset validation RNG for deterministic crops each epoch
        if seed is not None:
            val_crop.reset_rng(seed)

        val_loss = validate_one_epoch(
            model, val_loader, criterion, device, use_amp, amp_dtype
        )

        # LR scheduling
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        epochs_trained = epoch + 1

        # === Report intermediate value to Optuna for pruning ===
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise TrialPruned()

        # Early stopping
        if patience_counter >= patience:
            break


    # 5. (Optional) Full-frame evaluation for richer metrics

    if args.eval_on_full_frames and not args.smoke_test:
        model.eval()
        evaluator = evaluate_model_on_dataset(
            model, val_dataset, device, use_amp=use_amp, amp_dtype=amp_dtype
        )
        global_metrics = evaluator.get_global_metrics()

        trial.set_user_attr("val_f1", global_metrics["f1_score"])
        trial.set_user_attr("val_iou", global_metrics["iou"])
        trial.set_user_attr("val_precision", global_metrics["precision"])
        trial.set_user_attr("val_recall", global_metrics["recall"])

    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("epochs_trained", epochs_trained)

    return best_val_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for X-point classifier"
    )

    # --- Optuna settings ---
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="xpoint-tuning",
        help="Optuna study name (default: xpoint-tuning)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///optuna_xpoint.db",
        help="Optuna storage URL (default: sqlite:///optuna_xpoint.db)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Stop after this many seconds (default: None, run all trials)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "hyperband", "none"],
        help="Pruning strategy (default: median)",
    )

    # --- Data settings (same as XPointMLTest.py) ---
    parser.add_argument("--paramFile", type=Path, default=None)
    parser.add_argument("--xptCacheDir", type=Path, default=None)
    parser.add_argument("--trainFrameFirst", type=int, default=1)
    parser.add_argument("--trainFrameLast", type=int, default=140)
    parser.add_argument("--validationFrameFirst", type=int, default=141)
    parser.add_argument("--validationFrameLast", type=int, default=150)

    # --- Training settings (fixed across all trials) ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Max epochs PER TRIAL (default: 300, lower than full training for speed)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience per trial (default: 30)",
    )
    parser.add_argument("--use-amp", action="store_true", help="Enable AMP")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true")

    # --- Evaluation ---
    parser.add_argument(
        "--eval-on-full-frames",
        action="store_true",
        help="Run full-frame evaluation after each trial (slower but gives F1/IoU)",
    )

    # --- Output ---
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="./optuna_results",
        help="Directory for result files (default: ./optuna_results)",
    )

    # --- Testing ---
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 3 trials with synthetic data (no paramFile needed)",
    )

    return parser.parse_args()


def print_study_summary(study, results_dir):
    """Print and save a summary of the Optuna study results."""

    print("\n" + "=" * 70)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 70)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\nStudy name:          {study.study_name}")
    print(f"Total trials:        {len(study.trials)}")
    print(f"  Completed:         {len(completed)}")
    print(f"  Pruned:            {len(pruned)}")
    print(f"  Failed:            {len(failed)}")

    if not completed:
        print("\nNo completed trials to summarize.")
        return

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  Best validation loss: {best.value:.6f}")
    print(f"  Hyperparameters:")
    for key, value in sorted(best.params.items()):
        if isinstance(value, float):
            print(f"    {key:25s} {value:.6g}")
        else:
            print(f"    {key:25s} {value}")

    if best.user_attrs:
        print(f"  Additional metrics:")
        for key, value in sorted(best.user_attrs.items()):
            if isinstance(value, float):
                print(f"    {key:25s} {value:.4f}")
            else:
                print(f"    {key:25s} {value}")

    # --- Top 5 trials ---
    sorted_trials = sorted(completed, key=lambda t: t.value)
    print(f"\nTop 5 trials:")
    print(
        f"  {'#':>4s}  {'Val Loss':>10s}  {'LR':>10s}  {'WD':>10s}  "
        f"{'Drop':>6s}  {'BS':>4s}  {'Patch':>5s}  {'Ch':>4s}  {'Sched':>8s}"
    )
    for t in sorted_trials[:5]:
        p = t.params
        print(
            f"  {t.number:4d}  {t.value:10.6f}  "
            f"{p.get('learning_rate', 0):10.2e}  "
            f"{p.get('weight_decay', 0):10.2e}  "
            f"{p.get('dropout_rate', 0):6.3f}  "
            f"{p.get('batch_size', 0):4d}  "
            f"{p.get('patch_size', 0):5d}  "
            f"{p.get('base_channels', 0):4d}  "
            f"{p.get('scheduler', 'n/a'):>8s}"
        )

    print("=" * 70)

    # --- Save results to JSON ---
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "study_name": study.study_name,
        "n_trials_total": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
        "all_completed_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in sorted_trials
        ],
    }

    results_path = results_dir / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # --- Generate shell command for retraining with best params ---
    p = best.params
    cmd_lines = [
        "#!/bin/bash",
        f"# Auto-generated from Optuna study: {study.study_name}",
        f"# Best trial #{best.number} with val_loss={best.value:.6f}",
        "",
        "python -u ${rcRoot}/reconClassifier/XPointMLTest.py \\",
        f"  --learningRate {p.get('learning_rate', 5e-4):.6g} \\",
        f"  --weightDecay {p.get('weight_decay', 5e-5):.6g} \\",
        f"  --dropoutRate {p.get('dropout_rate', 0.15):.4g} \\",
        f"  --batchSize {p.get('batch_size', 64)} \\",
    ]

    sched = p.get("scheduler", "cosine")
    cmd_lines.append(f"  --scheduler {sched} \\")
    if sched == "plateau":
        cmd_lines.append(
            f"  --plateau-factor {p.get('plateau_factor', 0.5):.4g} \\"
        )
        cmd_lines.append(
            f"  --plateau-patience {p.get('plateau_patience', 5)} \\"
        )

    cmd_lines.extend(
        [
            "  --use-amp \\",
            "  --seed 42 \\",
            "  --require-gpu \\",
            "  --fixed-val-crops \\",
            "  --epochs 1200 \\",
            "  --patience 200 \\",
            "  --checkPointFrequency 200 \\",
            "  --paramFile=${PARAM_FILE} \\",
            "  --xptCacheDir=${CACHE_DIR}",
        ]
    )

    cmd_path = results_dir / "retrain_best_params.sh"
    with open(cmd_path, "w") as f:
        f.write("\n".join(cmd_lines) + "\n")
    os.chmod(cmd_path, 0o755)

    print(f"Retrain command saved to: {cmd_path}")
    print(f"\nTo retrain with best hyperparameters:")
    print(f"  bash {cmd_path}")
    print("=" * 70)


def main():
    args = parse_args()

    # --- Smoke test overrides ---
    if args.smoke_test:
        print("=" * 60)
        print("SMOKE TEST MODE: 3 trials with synthetic data")
        print("=" * 60)
        args.n_trials = 3
        args.epochs = 5
        args.patience = 3
        args.eval_on_full_frames = False

    # --- Validate args ---
    if not args.smoke_test:
        if args.paramFile is None:
            print("ERROR: --paramFile is required (or use --smoke-test)")
            sys.exit(1)
        if not args.paramFile.exists():
            print(f"ERROR: paramFile {args.paramFile} does not exist")
            sys.exit(1)

    if args.require_gpu and not torch.cuda.is_available():
        print("ERROR: --require-gpu set but no CUDA device found")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    # Pre-load datasets ONCE (expensive I/O — shared across all trials)

    if not args.smoke_test:
        print("\nLoading datasets (shared across all trials)...")
        t0 = time.time()
        train_dataset = XPointDataset(
            args.paramFile,
            range(args.trainFrameFirst, args.trainFrameLast),
            xptCacheDir=args.xptCacheDir,
            rotateAndReflect=False,
        )
        val_dataset = XPointDataset(
            args.paramFile,
            range(args.validationFrameFirst, args.validationFrameLast),
            xptCacheDir=args.xptCacheDir,
            rotateAndReflect=False,
        )
        print(f"Datasets loaded in {time.time() - t0:.1f}s")
        print(f"  Training frames:   {len(train_dataset)}")
        print(f"  Validation frames: {len(val_dataset)}")

        # Attach to args for access in objective()
        args._train_dataset = train_dataset
        args._val_dataset = val_dataset


    # Create Optuna study

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=5
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource=args.epochs, reduction_factor=3
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\nResuming study '{args.study_name}' with {n_existing} existing trials")
        if study.best_trial:
            print(f"Current best val_loss: {study.best_trial.value:.6f}")


    # Run optimization

    print(f"\nStarting Optuna optimization")
    print(f"  Trials:      {args.n_trials}")
    print(f"  DB:          {args.db}")
    print(f"  Pruner:      {args.pruner}")
    print(f"  Max epochs:  {args.epochs} per trial")
    print(f"  Patience:    {args.patience} per trial")
    if args.timeout:
        print(f"  Timeout:     {args.timeout}s")
    print()

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )


    # Results

    print_study_summary(study, args.results_dir)

    # --- Try to generate visualization plots ---
    try:
        from optuna.visualization import (
            plot_param_importances,
            plot_optimization_history,
            plot_parallel_coordinate,
        )

        fig_dir = args.results_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for name, plot_fn in [
            ("param_importances", plot_param_importances),
            ("optimization_history", plot_optimization_history),
            ("parallel_coordinate", plot_parallel_coordinate),
        ]:
            try:
                fig = plot_fn(study)
                fig.write_html(str(fig_dir / f"{name}.html"))
                print(f"  Saved: {fig_dir / name}.html")
            except Exception as e:
                print(f"  Skipped {name}: {e}")

    except ImportError:
        print("\nNote: pip install plotly for interactive visualizations")


if __name__ == "__main__":
    main()