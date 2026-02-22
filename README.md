# Simulation-Based Revenue Optimization  
Event-Driven Simulation with Statistical Modeling and Diagnostic Analytics

---

## Overview

This project implements an event-driven simulation engine for modeling operational performance across multiple business scenarios and applies advanced regression diagnostics to analyze revenue drivers.

The system integrates simulation, statistical inference, econometric diagnostics, regularization techniques, and visualization into a complete analytical pipeline.

The project demonstrates applied knowledge in:

- Operations Research  
- Statistical Inference  
- Econometrics  
- Machine Learning Stabilization  
- Simulation Modeling  
- Data Pipeline Engineering  

---

## System Architecture

Simulation Engine → Dataset Generator → Regression Pipeline → Diagnostics → Visualization → Web Dashboard

### Components

| Module | Responsibility |
|--------|---------------|
| SeaWorldSimulation.py | Event-driven simulation core |
| simulate.py | Multi-run dataset generator |
| analyze.py | OLS modeling and diagnostics |
| webapp.py | Interactive dashboard |
| SeaWorldLinearRegression.py | Legacy regression pipeline |
| requirements.txt | Dependencies |

---

## Simulation Engine

The system models multiple operational scenarios:

- BASE  
- ALT1–ALT10  

Each simulation run produces:

- avg_rating  
- avg_food_income  
- total_customers  
- total_revenue  
- additional operational metrics  

Features:

- Custom seed control  
- Multiple replications  
- Scenario-level investment parameters  
- ROI comparison  

---

## Regression Model

Primary model:

Total Revenue = β₀ + β₁(avg_rating) + β₂(avg_food_income) + β₃(total_customers)

Implemented using statsmodels OLS.

---

## Statistical Diagnostics Included

### Model Fit
- R-squared  
- Adjusted R-squared  
- F-statistic  

### Inference
- t-tests  
- p-values  
- Confidence intervals  
- Robust standard errors (HC3 support)  

### Residual Analysis
- Residuals vs Fitted  
- Q-Q Plot  
- Jarque–Bera normality test  

### Heteroskedasticity
- Breusch–Pagan test  

### Multicollinearity
- Condition number  
- Ridge regression stabilization  
- Automatic Ridge alpha selection  

### Influence Diagnostics
- Cook’s Distance  
- Leverage metrics  

---

## Scenario ROI Analysis

The system computes:

- Mean revenue per scenario  
- Investment cost comparison  
- ROI per dollar invested  
- Best-performing scenario identification  

---

## Web Dashboard

Launch locally:

```bash
python webapp.py
