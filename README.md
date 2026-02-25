# Simulation-Based Revenue Optimization  
Event-Driven Simulation with Statistical Modeling and Diagnostic Analytics

---

## Motivation

I built this project because I wanted to connect the pieces of my academic background into one coherent system.

In my studies, I encountered Monte Carlo simulation, regression analysis, statistical diagnostics, and optimization as separate subjects. Each course focused on a different tool. What interested me was not mastering each tool individually, but understanding how they interact in a real analytical workflow.

This project started from a simple question: what happens when we treat simulation not as the final result, but as the data-generating process for a deeper statistical analysis?

Instead of stopping at simulated averages, I wanted to generate structured datasets, estimate econometric models, validate assumptions, test numerical stability, and interpret the results through diagnostics. At the same time, I wanted to make the system interactive, so the experimentation process itself becomes transparent and reproducible.

The motivation was not to build “just another simulation” or “just another regression model.” It was to create a layered analytical environment where behavioral assumptions, uncertainty, and statistical validation exist within the same framework.

For me, this project represents the transition from solving exercises to designing systems.

---

## Overview

The platform implements a discrete-event simulation engine that models operational performance under multiple strategic alternatives. The simulated output is transformed into a structured dataset and analyzed using an econometric pipeline that evaluates:

- Coefficient magnitude and statistical significance  
- Multicollinearity  
- Residual behavior  
- Heteroskedasticity  
- Influence diagnostics  
- Stability under regularization  

The system supports 10+ configurable alternatives and can easily scale to additional scenarios or expanded regression specifications.

---

## System Architecture

Simulation Engine → Dataset Generator → Regression Pipeline → Diagnostics → Visualization → Web Dashboard

### Core Components

| Module | Responsibility |
|--------|---------------|
| SeaWorldSimulation.py | Event-driven Monte Carlo simulation engine |
| simulate.py | Multi-run dataset generation |
| analyze.py | OLS estimation and diagnostics |
| webapp.py | Interactive dashboard |
| SeaWorldLinearRegression.py | Legacy regression workflow |
| requirements.txt | Dependency management |

---

# Interface and Visualization

## Home Screen

![Home Screen](screenshots/simuhome.png)

Centralized control for simulation and modeling.

---

## Simulation Controls

![Simulation Controls](screenshots/simuhome2.png)

Clear separation between operational simulation and econometric configuration.

---

## Simulation Execution

![Simulation Loading](screenshots/loading_simu.png)

Transparent Monte Carlo progress tracking.

---

## Regression Output

![Regression Summary](screenshots/regression_summary.png)

Full OLS summary including:

- Coefficients  
- t-values  
- p-values  
- Confidence intervals  
- AIC/BIC  

---

## Extended Summary

![Regression Summary Extended](screenshots/regression_summary2.png)

Includes multicollinearity warnings and interpretive notes.

---

## Automated Analytical Commentary

![Analyst Chat](screenshots/analystchat.png)

Structured interpretation of:

- Normality  
- Heteroskedasticity  
- Multicollinearity  
- Ridge recommendations  

---

## Scenario Revenue Overview

![Scenario Revenue Overview](screenshots/scenario_revenue_overview.png)

ALT2 (Website + LargeTube investment) produces the highest mean revenue.

Structural improvements affecting demand flow and capacity yield greater returns than marginal efficiency changes.

The system supports dynamic comparison of 10+ alternatives and can scale further without architectural modification.

---

# Simulation Engine

## Mathematical Structure

The simulation models a stochastic theme-park-style ecosystem in which revenue emerges from visitor behavior.

Each visitor:

1. Arrives according to a probabilistic process  
2. Engages in service interactions  
3. Generates revenue components conditionally  
4. Contributes to satisfaction metrics  

Formally:

Total_Revenue = Σᵢ (Foodᵢ + Photoᵢ + Receptionᵢ)

Because revenue is structurally aggregated from visitor-level outcomes, operational metrics are inherently interdependent.

---

## Scenario Parameterization

The system includes BASE and ALT1–ALT10 configurations. Each alternative modifies structural parameters such as:

- Service efficiency multipliers  
- Marketing intensity  
- Conversion rates  
- Capacity expansion  
- Investment allocations  

The architecture allows adding more than 10 alternatives without structural changes to the analytical pipeline.

Scenarios represent shifts in the data-generating process itself, not merely categorical labels.

---

## Monte Carlo Replication

Each scenario is evaluated over multiple independent replications.

The system computes:

E[Revenue | Scenario]  
Var(Revenue | Scenario]

This enables comparison under stochastic uncertainty rather than deterministic output.

---

## Output Metrics

Each replication generates:

- avg_rating  
- avg_food_income  
- total_customers  
- total_revenue  
- auxiliary metrics  

These variables form the regression dataset. The regression specification itself can be expanded to include additional predictors or categorical encodings of alternatives.

---

# Econometric Modeling Framework

## Model Specification

The baseline OLS model:

Total_Revenue = β₀  
+ β₁(avg_rating)  
+ β₂(avg_food_income)  
+ β₃(total_customers)  
+ Scenario Dummies  
+ ε  

This framework supports:

- Adding additional regressors  
- Including interaction terms  
- Expanding scenario encoding  
- Modifying functional forms  

The model serves as a structural sensitivity analysis of the simulated revenue-generating mechanism.

---

# Statistical Analysis and Interpretation

## Structural Expectations

Because revenue is aggregated from behavioral variables, we expect:

- Strong explanatory power  
- High interdependence among predictors  
- Dominance of total_customers  
- Multicollinearity  

These expectations align with simulation design.

---

## Regression Output Overview

Below is the full OLS summary generated by `statsmodels`:

![Regression Summary](screenshots/regression_summary.png)

The table includes:

- Estimated coefficients (β̂)  
- Standard errors  
- t-statistics  
- p-values  
- 95% confidence intervals  
- R² and Adjusted R²  
- F-statistic and global significance test  
- AIC and BIC information criteria  
- Condition number (multicollinearity indicator)  

Key numerical results from the output:

- R² ≈ 0.999  
- Adjusted R² ≈ 0.999  
- F-statistic p-value ≈ 0.000  
- Condition number ≈ 2.3e+05  

These values form the basis for the interpretation below.

---

## Model Fit

The regression produces:

- R² ≈ 0.999  
- Adjusted R² ≈ 0.999  
- F-statistic p-value ≈ 0  

This confirms near-complete explanatory alignment between operational drivers and revenue outcomes within the simulated structure.

Because the data-generating process structurally defines revenue as a function of behavioral components, the extremely high explanatory power is consistent with the system design rather than accidental overfitting.

---

## Coefficient Analysis

### Total Customers

From the regression output:

- Coefficient is strongly positive  
- Large t-statistic  
- p-value < 0.001  

Interpretation: Revenue scales directly with volume, as expected from the aggregation structure. Since revenue is computed as a sum across visitors, this variable naturally dominates variation.

The narrow confidence interval indicates high estimation precision across Monte Carlo replications.

---

### Average Rating

- Positive coefficient  
- Statistically significant  
- Confidence interval does not include zero  

This suggests that quality improvements indirectly enhance revenue, potentially through correlated spending behavior or increased engagement intensity.

The sign aligns with structural assumptions embedded in the simulation parameters.

---

### Average Food Income

- Positive but weaker statistical strength relative to volume  

Interpretation: Behavioral heterogeneity across replications introduces variability in per-visitor spending. While food income contributes to revenue, its marginal impact is smaller compared to total volume.

This reflects micro-level behavioral randomness in the Monte Carlo process.

---

### Scenario Coefficients

Scenario dummy variables capture structural shifts relative to the baseline configuration.

- Statistically significant coefficients indicate meaningful performance differences.
- Negative baseline coefficients suggest that certain investment alternatives outperform the no-investment configuration.

Because scenarios alter underlying behavioral parameters (conversion, efficiency, capacity), their coefficients quantify structural differences in expected revenue.

The model supports comparison across 10+ alternatives and can incorporate additional scenario encodings.

---

## Multicollinearity

From the regression summary:

- Condition number ≈ 2.3e+05  

This indicates strong multicollinearity, which is expected because:

- Revenue is partially constructed from related predictors.
- Customer volume and per-visitor income are structurally interdependent.

To address this, ridge regression was applied:

- R² remains effectively unchanged.
- Coefficients shrink by approximately 15%.
- Numerical stability improves.

This confirms structural validity while reducing variance inflation.

---

## Residual Diagnostics

### Residuals vs Fitted

![Residuals vs Fitted](screenshots/residuals_vs_fitted.png)

This plot displays residual values on the vertical axis and fitted values on the horizontal axis.

What it shows:

- Random dispersion around zero  
- No visible curvature  
- No funnel-shaped variance pattern  
- Mild clustering into groups across fitted value ranges  

The apparent grouping pattern likely reflects the presence of multiple structural alternatives (BASE, ALT1–ALT10).  
Since each scenario modifies underlying behavioral parameters (conversion rates, efficiency multipliers, capacity adjustments), the fitted values naturally cluster by scenario. In other words, the regression is applied to pooled data generated from structurally different regimes.

Interpretation:

The linear specification appears appropriate overall, and there is no strong evidence of global heteroskedasticity or functional misspecification. However, the visible grouping suggests that residual structure may partially reflect scenario-level heterogeneity.

From a classical linear regression perspective, there are several standard ways to address or formally test this type of structure:

- Include explicit scenario dummy variables (already implemented in the model)  
- Allow interaction terms between scenario indicators and key predictors  
- Use heteroskedasticity-robust standard errors (e.g., HC3)  
- Apply clustered standard errors at the scenario level  
- Estimate separate regressions per scenario for structural comparison  
- Introduce hierarchical or mixed-effects models if treating scenarios as higher-level groups  

In this implementation, scenario dummies are included, and robustness checks (including ridge regularization and diagnostic testing) confirm that the overall specification remains stable.

Thus, while residual clustering reflects structural heterogeneity across alternatives, it does not invalidate the linear framework. Instead, it confirms that scenario shifts meaningfully influence fitted revenue levels, which is consistent with the simulation design.

---

### Q-Q Plot

![Q-Q Plot](screenshots/qqplot.png)

This graph compares standardized residuals to theoretical normal quantiles.

What it shows:

- Points closely follow the 45-degree reference line  
- Minor deviations only at the tails  

Jarque–Bera p ≈ 0.598 suggests acceptable normality of residuals.

This supports the validity of standard OLS inference assumptions.

---

### Diagnostic Dashboard

![Diagnostics Dashboard](screenshots/diagnostics.png)

The combined dashboard includes:

- Predicted vs Actual plot  
- Residuals vs Fitted  
- Q-Q plot  
- Influence (Cook’s Distance)  

Predicted vs Actual alignment along the 45-degree line confirms strong predictive coherence within the simulated structure.

Influence metrics show no extreme outliers driving estimation results.

---

## Influence Analysis

Cook’s distance values remain below critical thresholds.

No replication dominates estimation.

Leverage dispersion appears moderate.

The regression appears stable across simulation runs.

---

# Final Research-Oriented Conclusion

## Final Conclusion

When combining the simulation layer with the regression framework, the system behaves consistently with its structural design. Revenue emerges from behavioral mechanisms, and the econometric results reflect that internal logic rather than contradict it.

The main findings are:

1. **Revenue variation is primarily driven by customer volume.**  
   The coefficient on total_customers is strongly positive and statistically significant, confirming that scale effects dominate revenue formation.

2. **Satisfaction contributes positively to monetization.**  
   The positive and significant coefficient on avg_rating suggests that quality-related parameters indirectly enhance revenue outcomes.

3. **Scenario-level structural shifts materially alter expected performance.**  
   Differences across 10+ configurable alternatives demonstrate that investment strategies and operational parameter changes meaningfully propagate through the revenue-generating process.

4. **Multicollinearity is structurally inherent but manageable.**  
   Given the interdependence of behavioral predictors, multicollinearity is expected. Ridge regularization preserves explanatory power while improving numerical stability.

5. **Residual behavior supports linear adequacy within the simulated environment.**  
   Diagnostic plots and formal tests indicate stable residual dispersion, acceptable normality, and no dominant influential observations.

In practice, this system operates as a controlled Monte Carlo experimentation platform. It enables structured comparison across multiple strategic alternatives, supports flexible regression specifications with extendable predictors, and provides sensitivity analysis under stochastic variability.

It is intentionally not designed as a production forecasting engine. Its purpose is to evaluate alternatives rigorously, understand structural relationships, and test stability under uncertainty.

For me, the most meaningful outcome is not the near-perfect model fit, but the coherence between behavioral assumptions, statistical evidence, and diagnostic validation. That alignment is what makes the framework analytically sound.

---

# Personal Reflection

This project represents a point where theory stopped being abstract and started becoming operational.

Throughout my studies in Industrial Engineering and Management, I learned simulation, regression analysis, statistical inference, and optimization as separate tools. In this project, I deliberately chose to connect them into a single coherent system. Instead of solving a predefined exercise, I built an environment where assumptions, parameters, uncertainty, and diagnostics interact in a structured way.

Designing the simulation forced me to think carefully about how behavioral mechanisms translate into measurable outcomes. Implementing the regression layer required me to question not only whether the model “fits,” but whether it is numerically stable, statistically valid, and structurally interpretable. The diagnostics stage pushed me to evaluate the assumptions behind the numbers rather than accepting them at face value.

What I value most in this project is not the near-perfect R², but the process behind it: anticipating multicollinearity before seeing it, verifying residual behavior instead of assuming linearity, testing robustness with ridge regularization, and comparing alternatives under uncertainty rather than relying on deterministic outputs.

This system reflects how I approach analytical problems:
I prefer building controlled experimental environments, understanding the structure behind the data, and validating conclusions with discipline rather than intuition alone.

For me, this project is less about predicting revenue and more about demonstrating how simulation modeling and econometric reasoning can coexist within a rigorous, interpretable decision-support framework.
