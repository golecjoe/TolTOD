#!/bin/bash
#SBATCH --job-name=step1_TODprocessing
#SBATCH --output=step1_TODprocessing-%j.out
#SBATCH --time=96:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p toltec-cpu
set -euo pipefail


python -u cleanscript.py

