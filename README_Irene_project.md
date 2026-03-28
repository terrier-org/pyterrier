# Trie-Based Execution for Improving Efficiency of PyTerrier Experiments

This project extends PyTerrier with efficient trie-based execution and visualization for information retrieval experiments.

## Key Features

- **Radix Trie Implementation**: Efficient Radix Trie data structures (`RadixNode`, `RadixTree`) for representing and optimizing experiment pipelines.
- **Execution Structures**: New execution strategies, including `tee_execution` and `linear_execution`, implemented in `_exec_structure.py` for flexible and efficient pipeline execution.
- **Experiment Planning**: Enhanced plan selection logic in `experiment.py` to leverage trie-based and linear execution for optimal performance.
- **Visualization**: Interactive HTML visualizations of experiment execution plans using `radix_schematic` and `draw_radix_html_schematic` in `schematic.py`, with supporting changes in `data/schematic.css` and `data/schematic.js`.
- **Comprehensive Testing**: Tests performed can be found in `tests/test_radix_trie.py` covering string-based and general trie operations, including transformer support.

### Requirements
* Python 3.13
* Packages: listed in `requirements.txt` 
* Tested on Windows 11, Google Colab (GPU would increase efficiency)

There is no manual.md for this project as it is not a software.

## Build Instructions

1. **Install this version of PyTerrier**:
   ```bash
   pip install -e .
   ```
2. **Run tests**:
   ```bash
   pytest tests/test_radix_trie.py
   ```
3. **Usage**:
   - See the example notebook `examples/notebooks/pipelines.ipynb` for usage patterns and experiment code for empirical evaluation conducted.

## File Overview

- `pyterrier/_evaluation/_trie.py`: Radix Trie implementation.
- `pyterrier/_evaluation/_exec_structure.py`: Execution strategies (tree and linear).
- `pyterrier/_evaluation/_experiment.py`: Execution plan selection logic.
- `pyterrier/schematic.py`: HTML visualization functions.
- `pyterrier/data/schematic.css`, `pyterrier/data/schematic.js`: Visualization styling and logic.
- `tests/test_radix_trie.py`: Unit tests for trie and execution logic.
- `project-irene folder`: Has the very initial naive string prefix identification implementation.
