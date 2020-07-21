# Plan Arithmetic: Compositional Plan Vectors for Multi-Task Control

[Original repository](https://github.com/cdevin/cpv)

[Original project webpage](https://sites.google.com/berkeley.edu/compositionalplanvectors/home)

To install: clone the repository and run

```sh
poetry install
```

Add the repo to you pythonpath by running

```sh
export PYTHONPATH=[path/to/repo/]cpv-il/crafting:$PYTHONPATH
```

To generate training data for the crafting environment, run

```sh
cd [path/to/repo/]cpv-il/`
poetry run python crafting/scripts/collect_composite_trajectories.py
```

This may take a while.

Then, collect the evaluation trajectories by running

```sh
poetry run python crafting/scripts/collect_reference_trajectories.py
```

To train a CPV-Full model, run

```sh
poetry run python crafting/gridworld/algorithms/cpv_experiments.py -H -P
```

This script will print where the checkpoints are saved and where the tensorboard logs are saved.

To evaluate the model online, run

```sh
poetry run python crafting/scripts/run_model_multitask_tensorboard.py --model [path/to/checkpoints_dir] --tb [path/to/tensorboard_dir] --type V3
```

The code for grasper experiments is coming soon!
