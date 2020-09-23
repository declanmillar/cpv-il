# Plan Arithmetic: Compositional Plan Vectors for Multi-Task Control

This project is a [fork](https://github.com/cdevin/cpv) of the [project](https://sites.google.com/berkeley.edu/compositionalplanvectors/home) by Devin et al. on learning compositional plan vectors using imitation learning. Our goal is extend there approach to pure [reinforcement learning](https://github.com/lauradarcy/CPV_RL). We aim to use this repository both a benchmark by adapting it to use a standardized version of the original [environment](https://github.com/lauradarcy/gym-craftingworld) and to test improvements to learning speed using HPC.

## Usage

Create teh virtual environment

```sh
conda env create --file environment.yml
```

Activate it

```sh
conda activate cpv
```

Add the repo to you pythonpath by running

```sh
export PYTHONPATH=$HOME/Projects/cpv-il/crafting:$PYTHONPATH
```

To generate training data for the crafting environment, run

```sh
python crafting/scripts/collect_composite_trajectories.py
```

This may take a while.

Then, collect the evaluation trajectories by running

```sh
python crafting/scripts/collect_reference_trajectories.py
```

To train a CPV-Full model, run

```sh
python crafting/gridworld/algorithms/cpv_experiment.py -H -P
```

This script will print where the checkpoints are saved and where the tensorboard logs are saved.

To evaluate the model online, run

```sh
python crafting/scripts/run_model_multitask_tensorboard.py --model [path/to/checkpoints_dir] --tb [path/to/tensorboard_dir] --type V3
```
