# Wind Farm Load Estimation
[![DOI](https://zenodo.org/badge/933592104.svg)](https://doi.org/10.5281/zenodo.15446360)

This repository provides all necessary materials to reproduce the analyses and results presented in our paper, currently under review, on wind farm load estimation. The codebase includes data processing pipelines, analysis scripts, and visualization tools, along with the datasets used in our research.

## Dependencies
- Python version: 3.11.1
- This project uses a local copy of an OpenOA v.2.3 (https://github.com/NREL/OpenOA). Other dependencies are listed in the requirements.txt

## Usage

The analysis workflow is organized in Jupyter notebooks:
1. `01_proc_raw_data.ipynb`: Initial data processing and cleaning
2. `02_sampling.ipynb`: Data sampling procedures
3. `03_gpr_training.ipynb`: Gaussian Process Regression model training
4. `03_pce_training.ipynb`: Polynomial Chaos Expansion model training
5. `04_case_study.ipynb`: Application and evaluation of models in case study

## Data Structure

### data/
#### case_study/
- `predictions.pkl`: Stored model predictions for the case study analysis

#### models/
- `gpr_models.pickle`: Trained Gaussian Process Regression models
- `pce_models.pickle`: Trained Polynomial Chaos Expansion models

#### samples/
- `sample_set.npy`: Output of the 02_sampling.ipynb notebook.

#### scada/
- `farmdata_*`: Processed farmdata including all turbines, ready for application

#### simulation/
- `sample_sim_setup/`: A set of representative openfast files for one simulation case.
- `casematrix.csv`: Simulation case definitions. Transformed to .csv from `sample_set.npy`
- `surrogate_data.csv`: Processed 10min load variables by case number

#### turbines/
- `IEA-3.4-130-RWT/`: IEA reference wind turbine model files (https://github.com/IEAWindSystems/IEA-3.4-130-RWT)
- `Adapted RWT model/`: Above model with minor changes to better fit case study.

## Citation

[Add citation for paper]

## Contact
- Alexander Mönnig: alexander.moennig@alterric.com
- Ulrich Römer: u.roemer@tu-braunschweig.de
