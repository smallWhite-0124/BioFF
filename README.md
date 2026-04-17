BioFF
BioFF (Bioinformatics Forward-Forward) is a classification tool tailored for bioinformatics tasks (e.g., normal/tumor sample classification) based on the Forward-Forward algorithm, optimized for gene/protein expression data.
Installation

# Acknowledgements & Open Source Credits
The core implementation and code structure of this project are based on the open-source repository below. We sincerely thank the original author for their contributions:
- Repository Name: pytorch_forward_forwar

Author: mpezeshki
        
Repository URL: https://github.com/mpezeshki/pytorch_forward_forwar
Notes
This project is developed and improved based on the aforementioned open-source repository, following open-source protocols and respecting the original author's work.
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
