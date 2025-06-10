# reconClassifier
Magnetic reconnection classifier for Gkeyll simulations on 2D domains

## Python Environment setup

The following commands should be run once on `checkers` to create a virtual environment that has access to pytorch and `pgkyl` ([postgkyl](https://github.com/ammarhakim/postgkyl)).

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

`==== 313131 passed, 1 skipped in 6.10s ====`


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

## Resuming Development Work

The following commands should be run on `checkers` **every time you create a new shell** to resume work in the existing virtual environment.

```
cd nsfCssiMlClassifier
source envPyTorch.sh
source pgkyl/bin/activate
```
