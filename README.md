# A Dynamical Theory of LoRA in Continual Learning

This repository contains the code accompanying the paper **"A Dynamical Theory of LoRA in Continual Learning"**.

The code implements the theoretical and numerical experiments used to study the dynamics of Low-Rank Adaptation (LoRA) in continual learning. In particular, we analyze how low-rank parameterization affects task learning, forgetting, and transfer when a model is trained sequentially on two tasks.

## Repository structure

```text
.
├── ODEs_theory.py           # Deterministic ODE dynamics for the order parameters
├── networks.py              # Teacher–student models and continual-learning datasets
├── training.py              # SGD simulations
├── logger.py                # Experiment logging and result storage
├── utils.py                 # Utilities and initialization procedures
├── main.py                  # Main experiment script
└── parameters.json          # Experiment configuration
└── data_visualization.ipynb # Notebook for data visualization
```

## Configuration

Experiments are configured through `parameters.json`.

The main parameters include:

| Parameter          | Description                                  |
| ------------------ | -------------------------------------------- |
| `N`                | Input dimension                              |
| `K`                | Number of hidden neurons of the student      |
| `M`                | Number of teacher features                   |
| `L`                | LoRA rank                                    |
| `rho`              | Correlation between the two tasks            |
| `method`           | Training method                              |
| `scoring_method`   | Scoring procedure used for selective methods |
| `rows_to_freeze`   | Number of frozen rows when using the SDGM    |
| `alpha`            | Number of examples for each training task    |

## Running the experiments

The main script is `main.py`.

When `theory` is enabled in `parameters.json`, the script computes the deterministic dynamics of the order parameters through discretized ODEs using Euler discretization.

The resulting ODE trajectories are stored using `ExperimentLogger`.

To run the finite-dimensional SGD simulations, set `training` to True:

## Training methods

The code supports the training procedures studied in the paper, including:

* `standard`
* `LoRA`
* `standard + SDGM`
* `LoRA + SDGM`

## Datasets

The repository contains experiments for both synthetic teacher–student problems and real datasets.

The synthetic setting is implemented through:

```python
networks.Data_and_Teachers
```

while continual-learning experiments on MNIST, Fashion-MNIST, and CIFAR are implemented through the corresponding dataset classes in `networks.py`. 

## Results

Results are stored in the directory specified by:

```json
"path_to_res_folder": "/path/to/results/"
```

The experiment logger stores both metadata and the corresponding trajectories, allowing theoretical ODE solutions and finite-dimensional simulations to be compared using the same parameter configuration.
A small script showing how to import the datas to analyze them is located in `data_visualization.ipynb`.