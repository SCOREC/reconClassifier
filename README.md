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
- `--learningRate`: Learning rate for training (default: 1e-4)
- `--batchSize`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--minTrainingLoss`: Minimum reduction in training loss in orders of magnitude (default: 2, set to 0 to disable check)

### Data Configuration
- `--trainFrameFirst`: First frame number for training data (default: 1)
- `--trainFrameLast`: Last frame number (exclusive) for training data (default: 140)
- `--validationFrameFirst`: First frame number for validation data (default: 141)
- `--validationFrameLast`: Last frame number (exclusive) for validation data (default: 150)
- `--paramFile`: Path to the parameter txt file containing gkyl input data
- `--xptCacheDir`: Path to directory for caching X-point finder outputs

### Training Optimization
- `--use-amp`: Enable automatic mixed precision training for faster training on modern GPUs
- `--amp-dtype`: Data type for mixed precision (`float16` or `bfloat16`, default: `bfloat16`)
- `--patience`: Patience for early stopping (default: 15 epochs)

### Output and Monitoring
- `--plot`: Enable creation of figures showing ground truth and model-identified X-points
- `--plotDir`: Directory where figures are written (default: `./plots`)
- `--checkPointFrequency`: Number of epochs between model checkpoints (default: 10)

### Performance Benchmarking
- `--benchmark`: Enable performance benchmarking (tracks timing, throughput, GPU memory)
- `--benchmark-output`: Path to save benchmark results JSON file (default: `./benchmark_results.json`)
- `--eval-output`: Path to save evaluation metrics JSON file (default: `./evaluation_metrics.json`)

### Testing
- `--smoke-test`: Run minimal smoke test for CI (overrides other parameters for quick validation)

### Example with Advanced Options

For faster training with mixed precision and early stopping:

```bash
python -u ${rcRoot}/reconClassifier/XPointMLTest.py \
--paramFile=/path/to/params.txt \
--xptCacheDir=/path/to/cache \
--epochs 200 \
--learningRate 1e-4 \
--batchSize 16 \
--use-amp \
--amp-dtype bfloat16 \
--patience 20 \
--plot \
--trainFrameLast 100 \
--validationFrameLast 120
```

## Resuming Development Work

The following commands should be run on `checkers` **every time you create a new shell** to resume work in the existing virtual environment.

```
cd nsfCssiMlClassifier
source envPyTorch.sh
source pgkyl/bin/activate
```

## Model Evaluation Metrics

The model evaluation system measures how well the classifier identifies X-points (magnetic reconnection sites) by treating it as a pixel-level binary classification problem.

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

The model uses a 9×9 pixel expansion around X-points to account for localization uncertainty while still requiring accurate region identification.