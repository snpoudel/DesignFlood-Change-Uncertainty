#!/bin/bash
#SBATCH --job-name=sim-hymod            # Job name
#SBATCH --output=sim-hymod%j.log       # Output file name (%j expands to jobID)
#SBATCH --error=sim-hymod%j.log        # Error file name (%j expands to jobID)
#SBATCH --time=300:00:00                 # Time limit (HH:MM:SS)
#SBATCH --nodes=2                       #5 Number of nodes
#SBATCH --ntasks=98                    #394 Number of tasks (one for each job), if you don't know numner of tasks beforehand there are ways to make this input dynamic as well
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2G                        # Memory per CPU core (adjust as needed)

# Load necessary modules
# All modules are loaded inside the virtual environment so don't need to load here (check: pip list modules when virtual environment is loaded) 
module load python/3.11.5
# Activate your virtual environment if needed
source ~/pyenv-hmodel/bin/activate

# Run your Python script with mpi
mpirun python3 3.5simp_hymod.py
