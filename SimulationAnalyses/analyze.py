from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

DEFAULT_FEATURES = ["avg_rating", "avg_food_income", "total_customers"]
DEFAULT_TARGET = "total_revenue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SeaWorld simulation results with OLS regression.")
    parser.add_argument("--in", dest="input_path", type=Path, default=Path("data/simulation_results.csv"), help="Input CSV path")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--model-target", default=DEFAULT_TARGET, help="Target metric column")
    parser.add_argument("--features", help="Comma-separated feature list (defaults to avg_rating,avg_food_income,total_customers)")
    parser.add_argument("--scenario-as-dummies", action="store_true", default=True, help="Include scenario dummies (default: True)")
    parser.add_argument("--no-scenario-dummies", action="store_false", dest="scenario_as_dummies", help="Disable scenario dummy variables")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--robust-se", default=None, help="Robust covariance type (e.g., HC3)")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Ridge alpha for stabilization (optional)")
    parser.add_argument("--ridge-auto", action="store_true", help="Select ridge alpha via cross-validation")
    parser.add_argument("--boxcox-check", action="store_true", help="Run Box-Cox diagnostic on target")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def resolve_features(df: pd.DataFrame, raw_features: str | None) -> list[str]:
    if raw_features is None or not raw_features.strip():
        features = DEFAULT_FEATURES.copy()
    else:
        features = [f.strip() for f in raw_features.split(",") if f.strip()]
    return features


def build_design_matrix(
    df: pd.DataFrame,
    features: list[str],
    include_scenario: bool,
) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")

    X = df[features].copy()

    if include_scenario:
        if "scenario_name" not in df.columns:
            raise ValueError("scenario_name column missing for dummy encoding")
        dummies = pd.get_dummies(df["scenario_name"], prefix="scenario", drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X, has_constant="add")
    return X


def clean_data(df: pd.DataFrame, target: str, features: list[str], include_scenario: bool) -> pd.DataFrame:
    required = set(features + [target])
    if include_scenario:
        required.add("scenario_name")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    before = len(df)
    df_clean = df.dropna(subset=list(required))
    dropped = before - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} rows due to missing values")
    return df_clean


def coerce_numeric(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    y_numeric = pd.to_numeric(y, errors="coerce")
    mask = ~(X_numeric.isna().any(axis=1) | y_numeric.isna())
    dropped = len(X_numeric) - int(mask.sum())
    if dropped > 0:
        print(f"Dropped {dropped} rows due to non-numeric values after coercion")
    X_numeric = X_numeric.loc[mask].astype(float)
    y_numeric = y_numeric.loc[mask].astype(float)
    return X_numeric, y_numeric


def fit_model(X: pd.DataFrame, y: pd.Series, robust_se: str | None):
    model = sm.OLS(y, X)
    if robust_se:
        # Robust SE relaxes homoskedasticity assumption while keeping coefficient estimates unchanged.
        return model.fit(cov_type=robust_se)
    return model.fit()


def build_recommendations(results, robust_se: str | None) -> str:
    recs: list[str] = []
    if getattr(results, "condition_number", None) is not None:
        cond = float(results.condition_number)
        if cond > 1e4:
            recs.append(
                "- Condition number is large; check multicollinearity and consider removing or combining correlated features."
            )
    if robust_se is None:
        recs.append(
            "- Consider rerunning with robust SE (e.g., HC3) if heteroskedasticity is suspected."
        )
    recs.append(
        "- Collect more runs if estimates are unstable or p-values are near the threshold."
    )
    if not recs:
        return ""
    return "\nRecommendations:\n" + "\n".join(recs) + "\n"


def save_reports(results, outdir: Path, robust_se: str | None) -> tuple[Path, Path]:
    report_path = outdir / "regression_report.txt"
    coef_path = outdir / "coefficients.csv"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(results.summary().as_text())
        f.write(build_recommendations(results, robust_se))

    coef_table = results.summary2().tables[1]
    coef_table.to_csv(coef_path)

    print(f"Saved report: {report_path}")
    print(f"Saved coefficients: {coef_path}")
    return report_path, coef_path


def plot_residuals_vs_fitted(results, outdir: Path) -> Path:
    path = outdir / "residuals_vs_fitted.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results.fittedvalues, results.resid, alpha=0.7, edgecolors="k", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_qq(results, outdir: Path) -> Path:
    path = outdir / "qq_plot.png"
    fig = sm.qqplot(results.resid, line="45", fit=True)
    plt.title("Q-Q Plot")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_predicted_vs_actual(results, y: pd.Series, outdir: Path) -> Path:
    path = outdir / "predicted_vs_actual.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, results.fittedvalues, alpha=0.7, edgecolors="k", linewidth=0.3)
    min_val = min(y.min(), results.fittedvalues.min())
    max_val = max(y.max(), results.fittedvalues.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_influence(results, outdir: Path) -> Path:
    path = outdir / "influence_plot.png"
    influence = results.get_influence()
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(leverage, cooks, alpha=0.7, edgecolors="k", linewidth=0.3)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Influence (Leverage vs Cook's Distance)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_analysis(
    input_path: Path,
    outdir: Path,
    target: str,
    features: str | None,
    scenario_as_dummies: bool,
    robust_se: str | None,
    no_plots: bool,
) -> tuple[str, list[str]]:
    ensure_dir(outdir)

    df = load_data(input_path)
    feature_list = resolve_features(df, features)
    df = clean_data(df, target, feature_list, scenario_as_dummies)

    X = build_design_matrix(df, feature_list, scenario_as_dummies)
    y = df[target]
    X, y = coerce_numeric(X, y)

    results = fit_model(X, y, robust_se)
    report_path, coef_path = save_reports(results, outdir, robust_se)

    outputs = [str(report_path), str(coef_path)]
    if not no_plots:
        outputs.extend(
            [
                str(plot_residuals_vs_fitted(results, outdir)),
                str(plot_qq(results, outdir)),
                str(plot_predicted_vs_actual(results, y, outdir)),
                str(plot_influence(results, outdir)),
            ]
        )

    summary_text = results.summary().as_text() + build_recommendations(results, robust_se)
    return summary_text, outputs


def _compute_ridge_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float | None,
    auto: bool,
    ols_params: pd.Series,
) -> dict:
    X_ridge = X.drop(columns=["const"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ridge.values)
    if auto:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        ridge = RidgeCV(alphas=alphas, fit_intercept=True)
        ridge.fit(X_scaled, y.values)
        alpha = float(ridge.alpha_)
    else:
        alpha = float(alpha or 1.0)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_scaled, y.values)
    r2 = ridge.score(X_scaled, y.values)

    ridge_coefs = pd.Series(ridge.coef_, index=X_ridge.columns)
    ols_coefs = ols_params.drop(labels=["const"], errors="ignore")

    shared = ridge_coefs.index.intersection(ols_coefs.index)
    shrink = None
    if len(shared) > 0:
        ridge_norm = float((ridge_coefs[shared].abs().mean()))
        ols_norm = float((ols_coefs[shared].abs().mean()))
        if ols_norm > 0:
            shrink = 1.0 - (ridge_norm / ols_norm)

    return {
        "alpha": float(alpha),
        "r2": float(r2),
        "shrink": float(shrink) if shrink is not None else None,
    }


def compute_insights(
    input_path: Path,
    target: str,
    features: str | None,
    scenario_as_dummies: bool,
    robust_se: str | None,
    ridge_alpha: float | None = None,
    ridge_auto: bool = False,
    boxcox_check: bool = False,
) -> dict:
    df = load_data(input_path)
    feature_list = resolve_features(df, features)
    df = clean_data(df, target, feature_list, scenario_as_dummies)

    scenario_means = (
        df.groupby("scenario_name")[target]
        .mean()
        .sort_values(ascending=False)
    )
    best_scenario = scenario_means.index[0] if not scenario_means.empty else "-"
    best_value = float(scenario_means.iloc[0]) if not scenario_means.empty else 0.0

    X = build_design_matrix(df, feature_list, scenario_as_dummies)
    y = df[target]
    X, y = coerce_numeric(X, y)
    results = fit_model(X, y, robust_se)

    jb_stat, jb_p, skew, kurtosis = jarque_bera(results.resid)
    condition_number = float(getattr(results, "condition_number", 0.0))
    r2 = float(results.rsquared)
    adj_r2 = float(results.rsquared_adj)
    f_pvalue = float(results.f_pvalue) if results.f_pvalue is not None else 1.0

    bp_stat, bp_p, _, _ = het_breuschpagan(results.resid, results.model.exog)
    fitted = results.fittedvalues
    resid = results.resid
    corr = float(abs(pd.Series(resid).corr(pd.Series(fitted))))
    influence = results.get_influence()
    cooks = influence.cooks_distance[0]
    max_cook = float(cooks.max()) if len(cooks) else 0.0

    significant = [
        name for name, pval in results.pvalues.items()
        if name != "const" and float(pval) < 0.05
    ]

    model_ok = r2 >= 0.7 and f_pvalue < 0.05 and jb_p > 0.05

    insights = []
    insights.append(
        "Diagnostics: Residuals vs Fitted should look like random scatter; strong trends imply nonlinearity."
    )
    insights.append(
        f"Residual trend check (|corr(residuals,fitted)|={corr:.2f}): "
        + ("no strong trend detected." if corr < 0.2 else "possible pattern; consider nonlinear terms.")
    )
    insights.append(
        f"Q-Q plot: JB p={jb_p:.3g}, skew={skew:.2f}, kurtosis={kurtosis:.2f} "
        + ("suggests near-normal residuals." if jb_p > 0.05 else "suggests non-normal tails.")
    )
    insights.append(
        f"Heteroskedasticity: Breusch-Pagan p={bp_p:.3g} "
        + ("(no strong evidence)." if bp_p > 0.05 else "(variance likely non-constant; use robust SE).")
    )
    insights.append(
        f"Influence: max Cook's distance={max_cook:.3f} "
        + ("(no extreme points)." if max_cook < 1 else "(check influential points).")
    )

    insights.append(
        f"Best scenario by mean {target}: {best_scenario} (mean={best_value:,.0f})."
    )

    if boxcox_check:
        if (y <= 0).any():
            insights.append("Box-Cox check skipped: target has non-positive values.")
        else:
            y_bc, lam = boxcox(y)
            results_bc = fit_model(X, pd.Series(y_bc, index=y.index), robust_se)
            jb_stat_bc, jb_p_bc, _, _ = jarque_bera(results_bc.resid)
            r2_bc = float(results_bc.rsquared)
            insights.append(
                f"Box-Cox check: lambda={lam:.3f}, R²={r2_bc:.3f}, JB p={jb_p_bc:.3g}."
            )
            if jb_p_bc > jb_p and r2_bc >= r2 - 0.01:
                insights.append("Recommendation: Box-Cox transform could improve normality without hurting fit.")
            elif r2_bc > r2 + 0.01:
                insights.append("Recommendation: Box-Cox transform improves fit; consider a transformed model.")
            else:
                insights.append("Box-Cox check: no clear improvement over the original model.")

    if ridge_auto or (ridge_alpha is not None and ridge_alpha > 0):
        ridge_metrics = _compute_ridge_metrics(X, y, ridge_alpha, ridge_auto, results.params)
        shrink_text = (
            f", average shrinkage ~{ridge_metrics['shrink']:.0%}"
            if ridge_metrics["shrink"] is not None
            else ""
        )
        insights.append(
            f"Ridge (alpha={ridge_metrics['alpha']}) R²={ridge_metrics['r2']:.3f}{shrink_text}."
        )
        if condition_number > 1e4 and ridge_metrics["r2"] >= r2 - 0.02:
            insights.append("Recommendation: prefer ridge for stability with similar fit.")
    elif condition_number > 1e4:
        insights.append("Recommendation: try ridge regression to stabilize coefficients.")
    insights.append(
        f"Model fit: R²={r2:.3f}, Adj. R²={adj_r2:.3f}, F-test p={f_pvalue:.3g}."
    )
    insights.append(
        f"Residual normality (Jarque-Bera p)={jb_p:.3g}; "
        + ("normality looks reasonable." if jb_p > 0.05 else "normality may be violated.")
    )
    if condition_number > 1e4:
        insights.append(
            f"High condition number ({condition_number:.2g}) suggests multicollinearity; consider dropping or combining correlated features."
        )
    if significant:
        insights.append(
            f"Significant predictors (p<0.05): {', '.join(significant)}."
        )
    else:
        insights.append("No predictors are significant at p<0.05; consider adding features or more runs.")

    if model_ok:
        insights.append("Conclusion: regression looks valid for inference; keep the model and consider robust SE for safety.")
    else:
        insights.append("Conclusion: revisit the model (check residuals, add runs/features, or use robust SE).")

    return {
        "best_scenario": best_scenario,
        "best_value": best_value,
        "insights": insights,
        "scenario_means": scenario_means.to_dict(),
    }


def main() -> None:
    args = parse_args()

    summary, outputs = run_analysis(
        input_path=args.input_path,
        outdir=args.outdir,
        target=args.model_target,
        features=args.features,
        scenario_as_dummies=args.scenario_as_dummies,
        robust_se=args.robust_se,
        no_plots=args.no_plots,
    )

    print(summary)
    if args.ridge_alpha is not None or args.ridge_auto or args.boxcox_check:
        insights = compute_insights(
            input_path=args.input_path,
            target=args.model_target,
            features=args.features,
            scenario_as_dummies=args.scenario_as_dummies,
            robust_se=args.robust_se,
            ridge_alpha=args.ridge_alpha,
            ridge_auto=args.ridge_auto,
            boxcox_check=args.boxcox_check,
        )
        print("\nRidge Insights:")
        for item in insights["insights"]:
            if "Ridge" in item or "ridge" in item or "Box-Cox" in item:
                print(f"- {item}")
    if not args.no_plots:
        print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
