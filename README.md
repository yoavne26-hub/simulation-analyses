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

## Regression Base Model

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

## Interface Preview

### 1. Home Screen

![Home Screen](screenshots/simuhome.png)

The main dashboard provides centralized control for running simulations and executing regression analysis.  
Users can select scenarios, configure replication count, define seeds, and toggle advanced modeling options such as robust standard errors and ridge regularization.

---

### 2. Simulation Controls & Regression Panel

![Simulation Controls](screenshots/simuhome2.png)

The interface separates operational simulation controls from econometric configuration.  
This enforces a clean workflow: generate data → specify model → run diagnostics.

---

### 3. Simulation Execution State

![Simulation Loading](screenshots/loading_simu.png)

During execution, the system displays:

- Current scenario  
- Run index  
- Total runs  
- Estimated time remaining  
- Progress tracking  

This enables transparency during long Monte Carlo runs.

---

### 4. Scenario Revenue Overview

![Scenario Revenue Overview](screenshots/scenario_revenue_overview.png)

Aggregated scenario-level revenue comparison.  
Supports ROI evaluation and investment strategy comparison.

---

### 5. Regression Summary Output

![Regression Summary](screenshots/regression_summary.png)

Full OLS regression output including:

- Coefficients  
- Standard errors  
- t-statistics  
- p-values  
- Confidence intervals  
- Model fit statistics  
- Information criteria (AIC/BIC)  

---

### 6. Extended Regression Diagnostics

![Regression Summary Extended](screenshots/regression_summary2.png)

Includes multicollinearity indicators (Condition Number) and inference notes.

---

### 7. Residual Diagnostics

![Residuals vs Fitted](screenshots/residuals_vs_fitted.png)

Residual analysis verifies:

- Linearity  
- Homoscedasticity  
- Random error structure  

---

### 8. Q-Q Normality Check

![Q-Q Plot](images/qqplot.png)

Assesses residual normality using theoretical quantiles.

---

### 9. Full Diagnostic Dashboard

![Diagnostics Dashboard](screenshots/diagnostics.png)

Includes:

- Residuals vs Fitted  
- Q-Q Plot  
- Predicted vs Actual  
- Leverage vs Cook’s Distance  

---

### 10. Automated Analytical Insights

![Analyst Chat](screenshots/analystchat.png)

The system generates structured analytical commentary including:

- Normality tests  
- Heteroskedasticity checks  
- Multicollinearity warnings  
- Influence diagnostics  
- Ridge stabilization recommendations  

---

### 11. Extended Insight Output

![Analyst Chat Extended](screenshots/analystchat2.png)

Summarizes:

- Significant predictors  
- Model validity  
- Stability assessment  
- Final recommendation  
