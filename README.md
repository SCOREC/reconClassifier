# reconClassifier
Magnetic reconnection classifier for Gkeyll simulations on 2D domains

## Python Environment setup

The following command should be run once on `checkers` to create a virtual environment that has access to pytorch and `pgkyl` ([postgkyl](https://github.com/ammarhakim/postgkyl)).

Note, the `git clone` commands below use the ssh protocol.  Please check that you have ssh keys configured on GitHub.

```
mkdir nsfCssiMlClassifier
pushd nsfCssiMlClassifier

#create an script that loads the needed pytorch modules,
#sets PYTHONPATH, and rcRoot to the working directory
cat << EOF > envPyTorch.sh
export PYTHONPATH=\$PYTHONPATH:$PWD/pgkylFrontEnd
module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
module load gcc/13.2.0-4eahhas mpich/4.2.3-q4swqut
module load py-torch/2.5.1-un62ppx py-torchvision/0.20.1-hnf42ha
export rcRoot=$PWD
EOF

#load the modules
source envPyTorch.sh

#setup a virtual environment to install pgkyl into
python -m venv pgkyl
source pgkyl/bin/activate
git clone git@github.com:scorec/reconClassifier.git
git clone -b cws/scorec git@github.com:scorec/pgkylFrontEnd.git
pip install --upgrade pip
pip install torchvision
git clone git@github.com:ammarhakim/postgkyl.git
pushd postgkyl
pip install -e .[adios,test]
pytest  # all tests should pass
popd
```

If there were no problems, then the last line of output from the above commands will contain the following (with a slightly different time):

`==== 31 passed, 1 skipped in 6.10s ====`


## run classifier with cached x-point finder results


create a bash script to run a test with two epochs and a small subset of the
training data ('frames')

```
pushd ${rcRoot}
cat << EOF > runReconClass.sh
#!/bin/bash
export OMP_NUM_THREADS=10
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-10"

date=$(date '+%Y-%m-%d-%H-%M-%S')
mkdir testdir_${date}
pushd $_
python -u ${rcRoot}/reconClassifier/XPointMLTest.py \
--paramFile=/space/cwsmith/nsfCssiSpaceWeather2022/mlReconnection2025/1024Res_v0/pkpm_2d_turb_p2-params.txt \
--xptCacheDir=/space/cwsmith/nsfCssiSpaceWeather2022/mlReconnection2025/1024Res_v0/cache04082025 \
--epochs 2 \
--learningRate 1e-3 \
--batchSize 2 \
--trainFrameLast 2 \
--validationFrameLast 143
popd
EOF
chmod +x runReconClass.sh
```

run it

```
./runReconClass.sh
```

## Command Line Options

The classifier supports several command line options for training configuration:

### Training Parameters
- `--learningRate`: Learning rate for training (default: 1e-5)
- `--weightDecay`: Weight decay for L2 regularization (default: 5e-4)
- `--dropoutRate`: Dropout rate for regularization (default: 0.3)
- `--batchSize`: Batch size for training (default: 1)
- `--epochs`: Number of training epochs (default: 2000)
- `--minTrainingLoss`: Minimum reduction in training loss in orders of magnitude (default: 3, set to 0 to disable check)

### Architecture
- `--baseChannels`: Base number of channels in the UNet encoder (default: 64)

### Target Representation
- `--targetType`: Per-pixel training target, `binary` (default; a 9×9 block marking each X-point) or `gaussian` (a 2D Gaussian peak of unit height centered on each X-point, with overlapping peaks combined by elementwise maximum)
- `--gaussianSigma`: Width in pixels of the Gaussian peaks when `--targetType gaussian` (default: 3.0)

### Loss Function
- `--lossFunction`: Loss function, `dice` (default), `focal_dice` (combined focal + dice loss for class imbalance), or `heatmap_focal` (CornerNet/CenterNet-style penalty-reduced focal loss for Gaussian targets; a warning is printed if it is used with `--targetType binary`)
- `--focalAlpha`: Focal loss alpha, class balance weight (default: 0.75)
- `--focalGamma`: Focal loss gamma, focusing parameter (default: 2.0)
- `--focalDiceWeight`: Weight of the dice component in FocalDiceLoss (default: 0.5)
- `--heatmapAlpha`: `heatmap_focal` focusing exponent, which down-weights pixels the network already predicts well (default: 2.0)
- `--heatmapBeta`: `heatmap_focal` exponent on (1 − target) in the penalty-reduced negative term, so pixels close to an X-point are penalized less for high predictions (default: 4.0)

### Learning Rate Schedule
- `--warmupEpochs`: Number of linear warmup epochs before the main scheduler kicks in (default: 0)
- `--scheduler`: Learning rate scheduler, `cosine` (default) or `plateau`
- `--plateau-factor`: ReduceLROnPlateau factor (default: 0.5)
- `--plateau-patience`: ReduceLROnPlateau patience in epochs (default: 5)
- `--plateau-min-lr`: ReduceLROnPlateau minimum learning rate (default: 1e-6)

### Stochastic Weight Averaging
- `--swa`: Enable Stochastic Weight Averaging for better generalization
- `--swaStart`: Fraction of total epochs after which SWA begins (default: 0.75)

### Data Configuration
- `--trainFrameFirst`: First frame number for training data (default: 1)
- `--trainFrameLast`: Last frame number (exclusive) for training data (default: 140)
- `--validationFrameFirst`: First frame number for validation data (default: 141)
- `--validationFrameLast`: Last frame number (exclusive) for validation data (default: 150)
- `--paramFile`: Path to the parameter txt file containing gkyl input data
- `--xptCacheDir`: Path to directory for caching X-point finder outputs
- `--posRatio`: Target ratio of training patches containing at least one X-point (default: 0.5)
- `--fixed-val-crops`: Use deterministic validation crops each epoch for stable val loss (default: False)

### Training Optimization
- `--use-amp`: Enable automatic mixed precision training for faster training on modern GPUs
- `--amp-dtype`: Data type for mixed precision (`float16` or `bfloat16`, default: `bfloat16`)
- `--patience`: Patience for early stopping (default: 15 epochs)
- `--early-stop-min-delta`: Minimum improvement in validation loss to reset early stopping (default: 0.0)
- `--seed`: Random seed for reproducibility (default: None for non-deterministic)
- `--require-gpu`: Require GPU to be available, exit if not found

### Output and Monitoring
- `--plot`: Enable creation of figures showing ground truth and model-identified X-points
- `--plotDir`: Directory where figures are written (default: `./plots`)
- `--checkPointFrequency`: Number of epochs between model checkpoints (default: 100)

### Performance Benchmarking
- `--benchmark`: Enable performance benchmarking (tracks timing, throughput, GPU memory)
- `--benchmark-output`: Path to save benchmark results JSON file (default: `./benchmark_results.json`)
- `--eval-output`: Path to save evaluation metrics JSON file (default: `./evaluation_metrics.json`)

### Testing
- `--smoke-test`: Run minimal smoke test for CI (overrides other parameters for quick validation)

### Example with Advanced Options

For training with custom regularization and reproducibility:
```bash
python -u ${rcRoot}/reconClassifier/XPointMLTest.py \
--paramFile=/path/to/params.txt \
--xptCacheDir=/path/to/cache \
--epochs 200 \
--learningRate 1e-4 \
--weightDecay 1e-3 \
--dropoutRate 0.3 \
--batchSize 16 \
--use-amp \
--amp-dtype bfloat16 \
--patience 20 \
--seed 42 \
--require-gpu \
--plot \
--trainFrameLast 100 \
--validationFrameLast 120
```

To train the Gaussian heatmap model used for point-level detection (the hand-tuned configuration behind the current point-level results):
```bash
python -u ${rcRoot}/reconClassifier/XPointMLTest.py \
--paramFile=/path/to/params.txt \
--xptCacheDir=/path/to/cache \
--targetType gaussian \
--gaussianSigma 3.0 \
--lossFunction heatmap_focal \
--heatmapAlpha 2.0 \
--heatmapBeta 4.0 \
--baseChannels 64 \
--learningRate 0.00224265 \
--weightDecay 0.00253832 \
--dropoutRate 0.1784 \
--batchSize 256 \
--posRatio 0.4355 \
--warmupEpochs 10 \
--scheduler cosine \
--use-amp \
--epochs 1200 \
--patience 200 \
--fixed-val-crops \
--seed 42 \
--require-gpu
```

## Hyperparameter Tuning with Optuna

The `optuna_tuner.py` script automates hyperparameter search over the knobs above (base channels, dropout, weight decay, learning rate, positive ratio, focal/dice weighting, scheduler choice, SWA start). It uses a Tree-structured Parzen Estimator sampler and a Median Pruner that aborts unpromising runs early based on the validation F1 curve.

```
python -u ${rcRoot}/reconClassifier/optuna_tuner.py \
--paramFile=/path/to/params.txt \
--xptCacheDir=/path/to/cache \
--n-trials 50 \
--study-name xpoint-tuning \
--db sqlite:///optuna_xpoint.db
```

The SQLite database is created automatically on first run and reloaded on subsequent runs with the same `--study-name`, so a study can be resumed or extended without re-running completed trials.

### Tuning the Gaussian model for point-level F1

`optuna_tuner_gaussian.py` is the corresponding tuner for Gaussian heatmap targets. It searches over the same training knobs (learning rate, weight decay, dropout, batch size, patch size, base channels, positive ratio, scheduler), trains every trial with `heatmap_focal` loss, and **maximizes point-level F1** on the validation frames instead of minimizing validation loss. Every `--f1-interval` epochs it runs full-frame inference on the validation frames, extracts peaks, matches them to the ground-truth X-points, and reports F1 to the pruner.

The peak-extraction threshold only changes how the network's output is read, so it is not an Optuna search dimension (that would cost a full training run per threshold value). Instead, each F1 evaluation sweeps `--threshold-grid` over the same heatmaps and scores the trial at its best threshold. The winning value is stored as the trial's `best_threshold` user attribute and appears in the summary table.

```
python -u ${rcRoot}/reconClassifier/optuna_tuner_gaussian.py \
--paramFile=/path/to/params.txt \
--xptCacheDir=/path/to/cache \
--n-trials 50 \
--epochs 300 \
--study-name gauss-tune \
--db sqlite:///optuna_gauss.db \
--use-amp \
--require-gpu
```

Options specific to this tuner:
- `--gaussianSigma`: Gaussian target width in pixels, held fixed during the search (default: 3.0)
- `--heatmap-alpha`, `--heatmap-beta`: `heatmap_focal` exponents (defaults: 2.0 and 4.0)
- `--f1-interval`: Epochs between point-level F1 evaluations (default: 10)
- `--threshold-grid`: Comma-separated peak-extraction thresholds swept at each evaluation (default: `0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65`)
- `--match-radius`: Matching radius in pixels (default: 5.0)
- `--epochs`: Maximum epochs per trial (default: 300)

Both tuners default to the same `--study-name` and `--db`, but they optimize different objectives (validation loss versus point-level F1), so give the Gaussian tuner its own study to keep the results separate.

## Cross-regime Transfer Evaluation

The PKPM-trained model can be evaluated zero-shot on additional Gkeyll datasets: the 5-moment ("5M") and 10-moment ("10M") fluid simulations, and a second, independent PKPM run ("PKPMv2") with the same physical and numerical configuration but a different turbulence realization, which is useful as a held-out test set. Evaluation runs in two steps: first build the X-point cache for the dataset, then evaluate a checkpoint on it, either with the point-level pipeline (see [Point-level Evaluation](#point-level-evaluation)) or with the pixel-level `test_xpoint_transfer.py`.

### Building the X-point cache for 5M/10M/PKPMv2

`run_hessian_and_build_cache.py` is the only script that runs the deterministic Hessian X-point classifier; it writes the per-frame results as `.npy` files so the training and evaluation scripts only ever read from cache. Trying to train or evaluate on an uncached frame raises a clear error pointing back to this script. For each frame it also writes a ground-truth point list, `{N}_xpts.csv`, with `row,col,class` columns (`class` is `X`, `Omax`, or `Omin`), which the point-level scorer reads. `--dataset` accepts `5M`, `10M`, or `PKPMv2`, and `--start`/`--end` are inclusive.

```
python -u ${rcRoot}/reconClassifier/run_hessian_and_build_cache.py \
--dataset 5M \
--start 1 --end 150 \
--workers 30
```

The `RC_EXTRACT_DIR` and `RC_CACHE_BASE` environment variables override the default raw-data and cache directories; `predict_points.py` and `score_point_predictions.py` read the same variables (see [Point-level Evaluation](#point-level-evaluation)). Pointing `RC_EXTRACT_DIR` at a node-local ramdisk (e.g. `/dev/shm/$USER`) significantly accelerates cache construction on machines where the raw data lives on a slow shared filesystem.

Caches built before the CSV output existed can be given ground-truth point lists without re-running the classifier:

```
python -u ${rcRoot}/reconClassifier/backfill_xpts_csv.py /path/to/cache --output-dir /path/to/writable/dir
```

`--output-dir` is needed when the cache itself isn't writable (the original PKPM cache, for example, is owned by another user); `--overwrite` rewrites CSVs that already exist.

### Running transfer evaluation

`test_xpoint_transfer.py` evaluates a checkpoint on the in-domain dataset and each transfer dataset using the pixel-level metrics described under [Model Evaluation Metrics](#model-evaluation-metrics), writing per-dataset and combined results to `transfer_eval_results/`. The caches for every dataset it evaluates must exist before it is run. It is configured through environment variables:
- `BEST_MODEL_PATH`: Checkpoint to evaluate
- `IN_DOMAIN`: Dataset treated as the in-domain reference; the others become zero-shot targets (default: `PKPM`)
- `OUTPUT_TAG`: Suffix added to output filenames so separate runs don't overwrite each other
- `THRESHOLD`: Pixel probability threshold for the pixel-level metrics (default: 0.5)
- `EDGE_MARGIN`: Diagnostic border margin in pixels (default: 0)

```
BEST_MODEL_PATH=/path/to/checkpoints/best_model.pt \
python -u ${rcRoot}/reconClassifier/test_xpoint_transfer.py
```

## Point-level Evaluation

X-points are sparse (tens per 1024² frame), so the main evaluation works on detected points rather than pixels. It has two steps and works with any trained checkpoint:

1. **Predict.** `predict_points.py` runs the checkpoint on every cached frame of each dataset, reduces the confidence map to one peak per connected region above `--threshold`, and writes `<output-root>/<dataset>/{N}_xpts.csv` with `row,col,confidence` columns.
2. **Score.** `score_point_predictions.py` matches the predicted points one-to-one against the ground-truth `{N}_xpts.csv` files within `--radius` pixels (greedy, highest confidence first). A matched prediction is a true positive, an unmatched prediction is a false positive, and an unmatched ground-truth X-point is a false negative. It prints F1, precision, recall, TP, FP, and FN for each dataset.

```
python -u ${rcRoot}/reconClassifier/predict_points.py \
--checkpoint /path/to/checkpoints/best_model.pt \
--datasets PKPM 5M 10M PKPMv2 \
--output-root ./predictions \
--base-channels 64

python -u ${rcRoot}/reconClassifier/score_point_predictions.py \
--predictions-root ./predictions \
--datasets PKPM 5M 10M PKPMv2 \
--radius 5.0 \
--json-out ./predictions/score_summary.json
```

`predict_points.py` options:
- `--checkpoint`: Trained model weights (e.g. `checkpoints/best_model.pt`)
- `--datasets`: Any of `PKPM`, `5M`, `10M`, `PKPMv2` (default: `PKPM 5M 10M`)
- `--output-root`: Base directory for the per-dataset prediction CSVs
- `--threshold`: Minimum confidence for a peak to count as a detection (default: 0.3)
- `--max-components`: If a frame's thresholded confidence map splits into more regions than this, no peaks are returned for it, since that indicates an undertrained model rather than real detections (default: 20000)
- `--base-channels`: Must match the value the checkpoint was trained with (default: 64)

`score_point_predictions.py` options:
- `--predictions-root`: Directory with one subdirectory of prediction CSVs per dataset
- `--datasets`: Datasets to score (default: `PKPM 5M 10M`)
- `--radius`: Matching radius in pixels (default: 5.0)
- `--label`: Name shown in the summary table
- `--json-out`: Optional JSON file with the global results and a per-frame breakdown

**Held-out numbers.** Both scripts process every cached frame, so on a dataset the model was trained on, the printed F1 mixes training and held-out frames and will look better than held-out performance. Report held-out results from the validation or test frames only, for example by aggregating those frames' TP/FP/FN from the per-frame breakdown in `--json-out`.

**Choosing the threshold.** `--threshold` sets the operating point; it is not learned by the network. Raising it drops low-confidence detections, trading recall for precision, and F1 can be sensitive to it. Choose it on validation frames and report results on frames that were not used to choose it.

**Data locations.** Dataset paths are built from environment variables, so a launcher script can point the whole pipeline at a different environment by exporting them, just as the training launchers set `--paramFile` and `--xptCacheDir`. The defaults are the shared DeltaAI locations:

| Variable | Controls | Read by | Default |
|---|---|---|---|
| `RC_EXTRACT_DIR` | Raw 5M / 10M / PKPMv2 data and parameter files | `run_hessian_and_build_cache.py`, `predict_points.py` | `/work/nvme/bfim/ssridhar6/mlReconnection2025` |
| `RC_CACHE_BASE` | 5M / 10M / PKPMv2 caches and all ground-truth CSVs | all three scripts | `/work/nvme/bfim/ssridhar6/mlReconnection2025/cache` |
| `RC_PKPM_V0_ROOT` | Parameter file and cache of the original PKPM run | `predict_points.py` | `/work/nvme/bfim/cwsmith/mlReconnection2025/1024Res_v0` |

Ground truth for every dataset, including the original PKPM run, is read from `$RC_CACHE_BASE/<dataset>/{N}_xpts.csv`; use `backfill_xpts_csv.py --output-dir` to create these for a cache you can't write to.

## Resuming Development Work

The following commands should be run on `checkers` **every time you create a new shell** to resume work in the existing virtual environment.

```
cd nsfCssiMlClassifier
source envPyTorch.sh
source pgkyl/bin/activate
```

## Model Evaluation Metrics

During training, `XPointMLTest.py` measures how well the classifier identifies X-points (magnetic reconnection sites) by treating it as a pixel-level binary classification problem, and writes the metrics described here. These pixel metrics are most meaningful with `--targetType binary`; for the Gaussian heatmap model, use the [point-level evaluation](#point-level-evaluation), which is the metric used for reported results.

### Key Metrics

The evaluation outputs several metrics saved to JSON files:

- **Accuracy**: Overall pixel classification correctness (can be misleading due to class imbalance)
- **Precision**: Fraction of detected X-points that are correct (measures false alarm rate)
- **Recall**: Fraction of actual X-points that were found (measures miss rate)
- **F1 Score**: Harmonic mean of precision and recall (balanced performance metric)
- **IoU**: Intersection over Union - spatial overlap quality between predicted and actual X-point regions

### Understanding the Results

**Good performance indicators:**
- F1 Score > 0.8
- IoU > 0.5  
- Similar metrics between training and validation sets (no overfitting)
- Low standard deviation across frames (consistent performance)

**Warning signs:**
- Large gap between training and validation metrics (overfitting)
- High precision but low recall (too conservative, missing X-points)
- Low precision but high recall (too aggressive, many false alarms)
- High frame-to-frame variation (inconsistent detection)

### Output Files

After training, the model produces:
- `evaluation_metrics.json`: Validation set performance
- `train_evaluation_metrics.json`: Training set performance  
- Performance plots in the `plots/` directory showing:
  - Training history (loss curves)
  - Model predictions vs ground truth
  - True positives (green), false positives (red), false negatives (yellow)

### Physics Context

For reconnection studies:
- **High recall is critical**: Missing X-points means missing reconnection events
- **Precision affects analysis**: False positives corrupt downstream calculations
- **IoU indicates localization**: Poor IoU means inaccurate X-point positions

With `--targetType binary`, the target marks a 9×9 pixel block around each X-point to account for localization uncertainty while still requiring accurate region identification. With `--targetType gaussian`, each X-point is a Gaussian peak, and the localization tolerance is set by the point-level matching radius instead.