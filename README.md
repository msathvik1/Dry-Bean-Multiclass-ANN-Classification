# Dry Bean Variety Classification using Multiclass Neural Network
**Domain:** Agriculture / Smart Farming  
**Dataset:** [UCI Machine Learning Repository (ID 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)  
**Optimizer:** Stochastic Gradient Descent (SGD) with Momentum and Cosine Annealing

---

## Quickstart (VS Code)

This workspace is already pre-configured for VS Code:
1. Open this folder in VS Code (`File > Open Folder...` -> select `dry_bean_ann_project`).
2. Run the single, all-in-one script:
   - Press `Ctrl + F5` or click **Run and Debug** in the left sidebar and hit the green Play button on **"Python: Run Dry Bean All-In-One Script"**.
   - Or open terminal in VS Code and run:
     ```bash
     python dry_bean_classification_ann.py
     ```

## Project Files
- `dry_bean_classification_ann.py`: Complete, self-contained single Python file executing the entire pipeline and generating plots.
- `dry_bean_ann_notebook.ipynb`: Interactive Jupyter Notebook.
- `dry_bean.csv`: Cached dataset containing 13,611 records and 16 morphological features.
- `dry_bean_evaluation_plots.png`: High-resolution 6-panel evaluation figure (Convergence, Accuracy trajectory, and Normalized Confusion Matrices).
- `.vscode/`: Pre-configured interpreter path (`Anaconda Python 3.13`) and launch configurations.
