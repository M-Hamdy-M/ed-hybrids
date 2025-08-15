# Hybrid Training of Deep Neural Networks with Multiple Output Layers for Tabular Classification
**Paper:** [Pattern Recognition, 170 (2026) 112035](https://doi.org/10.1016/j.patcog.2025.112035) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320325006958)

## Overview
This repository accompanies the study on hybrid training schemes that combine backpropagation (BP) and closed-form solutions (CFS) within ensemble-deep (ed) architectures for tabular data classification. The work introduces:

- **edAS**: a layer-wise adaptive hybrid that selects BP vs. CFS per layer using validation-guided sampling and early stopping.
- **MO-MLP**: a post-training hybrid that augments a conventional MLP with additional CFS-trained output heads and aggregates them in a boosted fashion.

The codebase provides model implementations, dataset loaders for UCI benchmarks, and reproducible notebooks for tuning and analysis.

## Repository Structure (High Level)
- `Main.ipynb` — runs full tuning/evaluation over multiple datasets and model families; saves aggregated results.
- `Analysis.ipynb` — performs ranking tables, significance testing, and summary plots from saved results.
- `models/` — model families (hybrids and baselines) and utilities (e.g., early stopping).
- `tuning/bo/` — Bayesian optimization scaffolding, optimizable wrappers, and utilities.
- `data/DLoader/` — dataset loaders and **expected location for corrected UCI splits**.

## Dependencies
The repository requires the following Python packages:

- numpy  
- pandas  
- scikit-learn  
- torch  
- tqdm  
- matplotlib  
- IPython  
- autorank  
- scipy  
- ConfigSpace  
- smac  
- openml  
- easydict  
- fvcore  

## Datasets
### Corrected UCI Splits
This repository **does not include the datasets**. You must download the corrected UCI data splits from the following repository:

[https://github.com/lingping-fuzzy/UCI-data-correct-split/tree/main](https://github.com/lingping-fuzzy/UCI-data-correct-split/tree/main)

After downloading, place the datasets inside a directory called `UCIdata_Corrected` inside:

```
data/DLoader/
```

The resulting directory structure should look like:
```
project_root/
├── data/
│   └── DLoader/
│       ├── UCIdata_Corrected/
│       │   ├── abalone/
│       │   │   ├── abalone_py.dat
│       │   │   ├── folds_py.dat
│       │   │   ├── labels_py.dat
│       │   │   └── validation_folds_py.dat
│       │   ├── arrhythmia/
│       │   ├── bank/
│       │   └── ... (other datasets)
│       ├── LoadUCI.py
│       ├── __init__.py
│       └── ...
```
Each dataset folder contains the raw data and its corrected 4-fold split files. The notebooks expect this structure.

### Supported Benchmarks
The experiments cover 33 UCI datasets spanning domains such as medicine, vision, finance, and robotics.  

## Reproducing the Main Experiments
1. **Run Tuning & Evaluation**  
   Open `Main.ipynb` and execute all cells.  
   - Defines:
     - A list of **models** (edAS variants, MO-MLP variants, edRVFL, MLP/SNN baselines, etc.).
     - The **dataset** list (33 UCI datasets).
     - **tuning settings** (e.g., `n_trials=200`, `evaluation_reps=10`).  
   - Parallelizes runs (via `torch.multiprocessing`) and stores per-rep metrics (train/val/test) and training times into a multi-index DataFrame.  
   - Results are saved to `./results/eval_stats.csv` (or an experiment-specific path defined in the notebook).

2. **Run Statistical Analysis**  
   Open `Analysis.ipynb` and execute all cells.  
   - Loads the saved CSV, computes per-model accuracies, ranks, and performs pair-wise Wilcoxon signed-rank tests.  
   - Produces formatted comparison tables and summary statistics suitable for inclusion in reports and supplements.  

> **Note:** This analysis depends on the file `./results/eval_stats.csv` generated during the tuning and evaluation step in `Main.ipynb`. Make sure to run the main experiments first.

## Adding Your Own Model
You can integrate a custom classifier into both the **evaluation** and **Bayesian optimization (BO) tuning** pipelines. This involves (A) implementing a classification model that follows the base interface, and (B) optionally creating an “optimizable” wrapper for BO.

### A) Implement the Classification Model
- **Base class to extend:**  
  `models/types/model.py` → `ClassificationModel`
- **Required methods in your subclass:**
  - `_fit(self, X_train, y_train, X_val, y_val, **fitting_args)` — implement the full training loop.
  - `_predict(self, X)` — return class scores/logits as a NumPy array of shape `(n_samples, n_classes)`.
  - *(You usually **do not** override `_score` — `ClassificationModel` already implements it via accuracy.)*

    ### Minimal example:
    ```python
    from models.types.model import ClassificationModel
    class YourModel(ClassificationModel):
        def __init__(self, n_layers, n_nodes, epochs, learning_rate, seed=None, verbose=0):
            super().__init__(seed=seed, verbose=verbose)
            self.n_layers = n_layers
            self.n_nodes = n_nodes
            self.epochs = epochs
            self.learning_rate = learning_rate

        def _fit(self, X_train, y_train, X_val=None, y_val=None, **fitting_args):
            # training logic goes here
            pass

        def _predict(self, X):
            # prediction logic goes here (return class scores/logits)
            return y
    ```

> **Example reference:** See `models/perceptron/mlp.py::MLPModel` for a complete constructor that accepts all training hyperparameters, implements `_fit`/`_predict`, and provides `evaluate`/`measure_flops`.

### B) Make the Model Tunable (Optimizable Wrapper)
- **Base wrapper to extend:**  
  `tuning/bo/core/types/optimizable_model.py` → `OptimizableModel`
- **Where to place your wrapper:**  
  `tuning/bo/optimizables/your_model.py`
- **Minimal requirements in your wrapper:**
  - `__init__(...)` — call `super().__init__(YourModel, base_params, name, verbose)`.  
    - `base_params` holds fixed defaults (e.g., `device`, `epochs`, etc.) that are *not* searched.
  - **Search space:** implement either:
    - `_get_bounds(self) -> dict[str, CS.Hyperparameter]`, **or**
    - `_get_configuration_space(self) -> ConfigSpace.ConfigurationSpace`.

> **Example reference:** See `tuning/bo/optimizables/mlp.py::OptimizableMLP` for how to wire `base_params`, define `_get_bounds`, and use `self.dataset_loader` to adapt ranges.
    
### Minimal example:    
```python
from tuning.bo.core.types.optimizable_model import OptimizableModel
from models.custom.your_model import YourModel
import ConfigSpace as CS

class OptimizableYourModel(OptimizableModel):
    def __init__(self, name="YourModel", verbose=0):
        base_params = {"epochs": 10, "learning_rate": 1e-3}
        super().__init__(YourModel, base_params, name, verbose)

    def _get_bounds(self):
        return {
            "n_layers": CS.UniformIntegerHyperparameter("n_layers", lower=1, upper=5, default_value=2),
            "n_nodes": CS.UniformIntegerHyperparameter("n_nodes", lower=10, upper=100, default_value=32),
        }

```

### C) Expose and Run
- **Include it in the experiments:**  
  In `Main.ipynb`, add your `OptimizableYourModel(...)` instance to the `optimizables` list. The notebook’s loop will handle tuning, evaluation, and logging.



### BibTeX
If you find this code useful in your research, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S0031320325006958$0):
```bibtex
@article{HAMDY2026112035,
    title = {Hybrid training of deep neural networks with multiple output layers for tabular data classification},
    journal = {Pattern Recognition},
    volume = {170},
    pages = {112035},
    year = {2026},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2025.112035},
    url = {https://www.sciencedirect.com/science/article/pii/S0031320325006958},
    author = {Mohamed Hamdy and Abdulaziz Al-Ali and Ponnuthurai N. Suganthan and Hussein Aly},
}
```
## Contact

Thank you for your interest!  
For any queries, please don't hesitate to [contact us](mailto:mm1905748@qu.edu.qa).

