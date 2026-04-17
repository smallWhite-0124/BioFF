BioFF
BioFF (Bioinformatics Forward-Forward) is a classification tool tailored for bioinformatics tasks (e.g., normal/tumor sample classification) based on the Forward-Forward algorithm, optimized for gene/protein expression data.
Installation
bash
运行
# Source code installation
git clone https://github.com/smallWhite-0124/BioFF.git
cd bioff
pip install .
Quick Start
1. Data Preparation
Prepare two txt files (one sample per line, the last column is integer label):
good_samples.txt: Positive samples (label 0)
bad_samples.txt: Negative samples (label 1)
2. One-click Run
python
运行
from bioff import run_prediction

# Core call
model, results = run_prediction(
    good_path="good_samples.txt",
    bad_path="bad_samples.txt"
)

# Check accuracy
print("Test set accuracy:", results["accuracy"])
Notes
Input txt files must be 2D matrices with consistent feature numbers per line
Automatically adapts to CPU/GPU, no manual configuration required
License
MIT