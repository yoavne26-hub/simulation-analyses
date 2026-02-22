# SeaWorld Simulation and Regression Project

This project contains an event-driven simulation and a regression analysis pipeline.

## Quick start (fast default)

Run a quick simulation (BASE/ALT1/ALT2, 5 runs) and write data:

```bash
python simulate.py
```

Analyze the generated data without plots:

```bash
python analyze.py --in data/simulation_results.csv --no-plots
```

## Full run

Run all scenarios with 30 replications:

```bash
python simulate.py --full
```

Analyze with plots saved to `outputs/`:

```bash
python analyze.py --in data/simulation_results.csv
```

## Web dashboard (local)

Launch a local dashboard that lets you run simulations and view plots:

```bash
python webapp.py
```

Then open: `http://127.0.0.1:8000`

## Generated files

- `data/simulation_results.csv` (simulation dataset)
- `outputs/regression_report.txt` (statsmodels summary)
- `outputs/coefficients.csv` (coefficient table)
- `outputs/*.png` (diagnostic plots, if not `--no-plots`)

## Common options

Change scenarios/runs/seed:

```bash
python simulate.py --scenarios BASE,ALT1,ALT2 --runs 10 --seed 123
```

Set regression target/features and robust SE:

```bash
python analyze.py --in data/simulation_results.csv --model-target total_revenue --features avg_rating,avg_food_income,total_customers --robust-se HC3
```

## Notes

- The analysis uses statsmodels OLS; robust standard errors (e.g., HC3) relax the homoskedasticity assumption while keeping coefficient estimates unchanged.
- `SeaWorldLinearRegression.py` is retained for backward compatibility but is deprecated in favor of `simulate.py` + `analyze.py`.
