# Changepoint Detection of Coverage Bias

This repository contains the code accompanying a research project on detecting
structural changes in global media coverage using changepoint detection methods.

The project uses multivariate time-series signals derived from GDELT and compares
several deep-learning approaches with a classical statistical baseline. The
repository includes code for data preprocessing, model training and inference,
changepoint analysis, and evaluation.

## Methods

The evaluated approaches include:

- TIRE
- TS-CP2
- SN-TS2Vec
- CorD-CPD
- Ruptures (statistical baseline)

Evaluation includes comparisons between model outputs, synthetic changepoint
experiments, and analysis of detected changes in relation to real-world events.

## Data

The analysis is based on data from the GDELT project. Raw GDELT data are not
included in this repository.

## Repository Structure

The repository contains scripts and notebooks for preprocessing, model
experiments, evaluation, and visualization. Further methodological details are
provided in the accompanying paper.
