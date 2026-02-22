"""
Multiple Linear Regression Analysis for SeaWorld Simulation
Runs simulation multiple times and performs regression analysis on metrics
with statistical hypothesis testing (alpha = 0.05)

Model: Total Revenue = f(avg_rating, avg_food_income, total_customers)
"""

import numpy as np
import random
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress, t as t_dist
from sklearn.linear_model import LinearRegression
from SeaWorldSimulation import (
    run_single_scenario,
    run_scenario_replications,
    SCENARIOS,
    SEED,
)

# Statistical parameters
ALPHA = 0.05
INITIAL_RUNS = 30
BASE_SEED = SEED
RELATIVE_ACCURACY = 0.1


class LinearRegressionAnalyzer:
    """Performs multiple linear regression and statistical analysis on simulation results."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        # Predictor variables (X)
        self.predictors = ["avg_rating", "avg_food_income", "total_customers"]
        # Target variable (Y)
        self.target = "total_revenue"
        self.scenarios = ["BASE", "ALT1", "ALT2", "ALT3", "ALT4", "ALT5", "ALT6", "ALT7", "ALT8", "ALT9", "ALT10"]
        
        # Store raw data
        self.all_results = {scn: {} for scn in self.scenarios}
        self.X = None  # Design matrix
        self.y = None  # Target vector
        self.regression_results = {}

    def run_simulations(self, num_runs: int, base_seed: int = SEED):
        """Run simulations for all scenarios and collect results."""
        print(f"Running {num_runs} simulation iterations for each scenario...")
        
        for scn_key in self.scenarios:
            print(f"  Running {scn_key} scenario...")
            result = run_scenario_replications(scn_key, num_runs, base_seed)
            
            # Store all predictor variables
            self.all_results[scn_key]["avg_rating"] = result["avg_rating"]
            self.all_results[scn_key]["avg_food_income"] = result["avg_food_income"]
            self.all_results[scn_key]["total_customers"] = result["total_customers"]
            self.all_results[scn_key]["total_revenue"] = result["total_revenue"]

    def prepare_regression_data(self):
        """Prepare data for multiple linear regression analysis.
        
        Collects predictor variables (avg_rating, avg_food_income, total_customers)
        and target variable (total_revenue).
        """
        X_values = []  # Design matrix
        y_values = []  # Target values
        
        # Collect all observations from all scenarios
        for scn_key in self.scenarios:
            num_runs = len(self.all_results[scn_key]["total_revenue"])
            
            for i in range(num_runs):
                x_row = [
                    float(self.all_results[scn_key]["avg_rating"][i]),
                    float(self.all_results[scn_key]["avg_food_income"][i]),
                    float(self.all_results[scn_key]["total_customers"][i]),
                ]
                X_values.append(x_row)
                y_values.append(float(self.all_results[scn_key]["total_revenue"][i]))
        
        self.X = np.array(X_values)
        self.y = np.array(y_values)
        
        return self.X, self.y

    def perform_multiple_regression(self):
        """Perform multiple linear regression with statistical tests."""
        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Call prepare_regression_data first.")
        
        # Remove NaN values
        mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
        X_clean = self.X[mask]
        y_clean = self.y[mask]
        
        if len(X_clean) <= len(self.predictors) + 1:
            raise ValueError("Insufficient data for regression analysis")
        
        # Fit the model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Predictions and residuals
        y_pred = model.predict(X_clean)
        residuals = y_clean - y_pred
        
        # Calculate statistics
        n = len(X_clean)
        p = len(self.predictors)  # number of predictors
        dof_residual = n - p - 1  # residual degrees of freedom
        if dof_residual <= 0:
            raise ValueError("Insufficient degrees of freedom for regression statistics")
        
        # Sum of squared errors
        sse = np.sum(residuals ** 2)
        mse = sse / dof_residual if dof_residual > 0 else 0
        rmse = np.sqrt(mse)
        
        # R-squared and adjusted R-squared
        ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (sse / ss_total) if ss_total > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof_residual
        
        # Standard errors of coefficients
        X_with_intercept = np.column_stack([np.ones(n), X_clean])
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_coefficients = np.sqrt(np.diag(XtX_inv) * mse)
        except np.linalg.LinAlgError:
            se_coefficients = np.full(p + 1, np.nan)
        
        # t-statistics and p-values
        coefficients = np.concatenate([[model.intercept_], model.coef_])
        t_stats = coefficients / se_coefficients
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), dof_residual))
        
        # Confidence intervals
        t_crit = t_dist.ppf(1 - self.alpha / 2, dof_residual)
        ci_lower = coefficients - t_crit * se_coefficients
        ci_upper = coefficients + t_crit * se_coefficients

        # Global F-test for model significance
        ss_reg = ss_total - sse
        ms_reg = ss_reg / p if p > 0 else 0
        f_stat = ms_reg / mse if mse > 0 else 0
        f_p_value = 1 - stats.f.cdf(f_stat, p, dof_residual) if p > 0 else 1.0

        # Residual normality (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(residuals) if n >= 3 else (np.nan, np.nan)

        # Breusch-Pagan test for heteroskedasticity
        bp_lm_stat, bp_p_value = self._breusch_pagan_test(X_clean, residuals)

        # VIF for multicollinearity
        vifs = self._calculate_vif(X_clean)
        
        return {
            "n": n,
            "p": p,
            "intercept": model.intercept_,
            "coefficients": model.coef_,
            "se_intercept": se_coefficients[0],
            "se_coefficients": se_coefficients[1:],
            "t_intercept": t_stats[0],
            "t_coefficients": t_stats[1:],
            "p_intercept": p_values[0],
            "p_coefficients": p_values[1:],
            "ci_intercept": (ci_lower[0], ci_upper[0]),
            "ci_coefficients": list(zip(ci_lower[1:], ci_upper[1:])),
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "rmse": rmse,
            "residuals": residuals,
            "y_pred": y_pred,
            "dof_residual": dof_residual,
            "f_stat": f_stat,
            "f_p_value": f_p_value,
            "shapiro_stat": shapiro_stat,
            "shapiro_p_value": shapiro_p,
            "bp_lm_stat": bp_lm_stat,
            "bp_p_value": bp_p_value,
            "vif": vifs,
        }

    def analyze_regression(self):
        """Perform and display multiple regression analysis."""
        print("\n" + "="*80)
        print("MULTIPLE LINEAR REGRESSION ANALYSIS")
        print("="*80)
        
        result = self.perform_multiple_regression()
        self.regression_results = result
        
        print(f"\nSample Size: {result['n']}")
        print(f"Number of Predictors: {result['p']}")
        print(f"Residual Degrees of Freedom: {result['dof_residual']}")
        
        print(f"\nRegression Equation:")
        print(f"Total Revenue = {result['intercept']:.2f}", end="")
        for i, pred_name in enumerate(self.predictors):
            coef = result['coefficients'][i]
            sign = "+" if coef >= 0 else "-"
            print(f" {sign} {abs(coef):.6f}*{pred_name}", end="")
        print()
        
        print(f"\nIntercept:")
        print(f"  Estimate: {result['intercept']:.6f}")
        print(f"  Std. Error: {result['se_intercept']:.6f}")
        print(f"  t-statistic: {result['t_intercept']:.6f}")
        print(f"  p-value: {result['p_intercept']:.6f}")
        print(f"  95% CI: ({result['ci_intercept'][0]:.6f}, {result['ci_intercept'][1]:.6f})")
        
        print(f"\nRegression Coefficients:")
        for i, pred_name in enumerate(self.predictors):
            coef = result['coefficients'][i]
            se = result['se_coefficients'][i]
            t_stat = result['t_coefficients'][i]
            p_val = result['p_coefficients'][i]
            ci = result['ci_coefficients'][i]
            
            print(f"\n  {pred_name}:")
            print(f"    Estimate: {coef:.6f}")
            print(f"    Std. Error: {se:.6f}")
            print(f"    t-statistic: {t_stat:.6f}")
            print(f"    p-value: {p_val:.6f}")
            print(f"    95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")
            
            is_sig = p_val < self.alpha
            print(f"    Result: {'*** SIGNIFICANT ***' if is_sig else 'NOT significant'} (α = {self.alpha})")
        
        print(f"\nModel Fit:")
        print(f"  R-squared: {result['r_squared']:.6f}")
        print(f"  Adjusted R-squared: {result['adj_r_squared']:.6f}")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  F-statistic: {result['f_stat']:.6f}")
        print(f"  F-test p-value: {result['f_p_value']:.6f}")
        print(f"  Shapiro-Wilk p-value: {result['shapiro_p_value']:.6f}")
        print(f"  Breusch-Pagan p-value: {result['bp_p_value']:.6f}")
        print(f"\nMulticollinearity (VIF):")
        for name, vif in zip(self.predictors, result["vif"]):
            print(f"  {name}: {vif:.6f}")
        print(f"\nInterpretation:")
        print(f"  - R² = {result['r_squared']:.4f} means {result['r_squared']*100:.2f}% of revenue variation")
        print(f"    is explained by these three predictors.")
        print(f"  - *** indicates p-value < {self.alpha}")

    def print_summary(self):
        """Print comprehensive summary of analysis."""
        print("\n" + "="*80)
        print("SUMMARY OF MULTIPLE LINEAR REGRESSION ANALYSIS")
        print("="*80)
        print(f"Significance level (alpha): {self.alpha}")
        print(f"Total simulations per scenario: {len(self.all_results['BASE']['total_revenue'])}")
        print(f"Total observations: {len(self.y)}")
        
        print("\nInterpretation Guide:")
        print("  - Positive coefficient: Metric increases revenue")
        print("  - Negative coefficient: Metric decreases revenue")
        print("  - p-value < alpha: Statistically significant predictor")
        print("  - p-value ≥ alpha: Not a significant predictor")
        print("  - R² closer to 1: Better model fit")

    @staticmethod
    def _ols_fit(X: np.ndarray, y: np.ndarray):
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.full(X_with_intercept.shape[1], np.nan)
        y_hat = X_with_intercept @ beta
        residuals = y - y_hat
        sse = float(np.sum(residuals ** 2))
        return beta, y_hat, residuals, sse

    @classmethod
    def _calculate_vif(cls, X: np.ndarray) -> list[float]:
        vifs = []
        for i in range(X.shape[1]):
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            _, y_hat, _, _ = cls._ols_fit(X_others, y_i)
            ss_total = np.sum((y_i - np.mean(y_i)) ** 2)
            ss_res = np.sum((y_i - y_hat) ** 2)
            r2 = 1 - (ss_res / ss_total) if ss_total > 0 else 0.0
            if r2 >= 1.0:
                vifs.append(float("inf"))
            else:
                vifs.append(1.0 / (1.0 - r2))
        return vifs

    @classmethod
    def _breusch_pagan_test(cls, X: np.ndarray, residuals: np.ndarray):
        if len(X) <= 2:
            return np.nan, np.nan
        y = residuals ** 2
        _, y_hat, _, _ = cls._ols_fit(X, y)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        r2 = 1 - (ss_res / ss_total) if ss_total > 0 else 0.0
        lm_stat = float(len(X) * r2)
        p_value = 1 - stats.chi2.cdf(lm_stat, X.shape[1])
        return lm_stat, p_value

    def generate_analytics_plots(self):
        """Generate comprehensive visualization of regression analysis."""
        result = self.regression_results
        y_pred = result["y_pred"]
        residuals = result["residuals"]
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.y, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        min_val = min(self.y.min(), y_pred.min())
        max_val = max(self.y.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Total Revenue ($)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Total Revenue ($)', fontsize=11, fontweight='bold')
        ax1.set_title('Actual vs Predicted Revenue', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Fitted
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Fitted Values ($)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q Plot (Normality test)
        ax4 = plt.subplot(2, 3, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Coefficient Plot with Confidence Intervals
        ax5 = plt.subplot(2, 3, 5)
        coef_names = ['Intercept'] + self.predictors
        coef_vals = np.concatenate([[result['intercept']], result['coefficients']])
        se_vals = np.concatenate([[result['se_intercept']], result['se_coefficients']])
        
        ci_lower = coef_vals - 1.96 * se_vals
        ci_upper = coef_vals + 1.96 * se_vals
        
        y_pos = np.arange(len(coef_names))
        colors = ['green' if result['p_coefficients'][i] < self.alpha else 'gray' for i in range(len(self.predictors))]
        colors = ['green'] + colors if result['p_intercept'] < self.alpha else ['gray'] + colors
        
        ax5.barh(y_pos, coef_vals, xerr=[coef_vals - ci_lower, ci_upper - coef_vals], 
                 capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(coef_names, fontsize=10)
        ax5.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
        ax5.set_title('Regression Coefficients with 95% CI', fontsize=12, fontweight='bold')
        ax5.axvline(x=0, color='r', linestyle='--', lw=2)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Significant (p < 0.05)'),
                          Patch(facecolor='gray', alpha=0.7, label='Not Significant')]
        ax5.legend(handles=legend_elements, loc='best', fontsize=9)
        
        # 6. R-squared and Model Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
        MODEL STATISTICS
        
        R² = {result['r_squared']:.6f}
        Adjusted R² = {result['adj_r_squared']:.6f}
        RMSE = ${result['rmse']:.2f}
        
        Sample Size (n) = {result['n']}
        Predictors (p) = {result['p']}
        Residual DOF = {result['dof_residual']}
        
        SIGNIFICANT PREDICTORS:
        """
        
        for i, pred in enumerate(self.predictors):
            if result['p_coefficients'][i] < self.alpha:
                stats_text += f"\n  ✓ {pred} (p={result['p_coefficients'][i]:.4f})"
        
        if not any(result['p_coefficients'] < self.alpha):
            stats_text += "\n  None"
        
        ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comprehensive regression plot saved as 'regression_analysis.png'")
        plt.show()

    def generate_coefficient_significance_plot(self):
        """Generate plot showing p-values and significance levels."""
        result = self.regression_results
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pred_names = self.predictors
        p_values = result['p_coefficients']
        colors = ['red' if p < self.alpha else 'blue' for p in p_values]
        
        y_pos = np.arange(len(pred_names))
        ax.barh(y_pos, p_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(x=self.alpha, color='green', linestyle='--', lw=2.5, label=f'α = {self.alpha}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pred_names, fontsize=11)
        ax.set_xlabel('p-value', fontsize=12, fontweight='bold')
        ax.set_title('Predictor Significance (p-values)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, p in enumerate(p_values):
            ax.text(p + 0.01, i, f'{p:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('coefficient_significance.png', dpi=300, bbox_inches='tight')
        print("✓ Coefficient significance plot saved as 'coefficient_significance.png'")
        plt.show()

    def generate_predictor_correlation_heatmap(self):
        """Generate correlation matrix heatmap of predictors."""
        import matplotlib.patches as mpatches
        
        # Create correlation matrix
        data_matrix = np.column_stack([self.X, self.y])
        col_names = self.predictors + ['Total Revenue']
        corr_matrix = np.corrcoef(data_matrix.T)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_names)))
        ax.set_yticks(np.arange(len(col_names)))
        ax.set_xticklabels(col_names, fontsize=10)
        ax.set_yticklabels(col_names, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values as text
        for i in range(len(col_names)):
            for j in range(len(col_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        ax.set_title('Correlation Matrix Heatmap', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")
        plt.show()

    def generate_scenario_comparison_plot(self):
        """Generate plots comparing revenue across scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Revenue by Scenario
        ax1 = axes[0, 0]
        scenario_names = []
        scenario_revenues = []
        
        for scn in self.scenarios:
            scenario_names.append(scn)
            scenario_revenues.append(self.all_results[scn]['total_revenue'])
        
        ax1.boxplot(scenario_revenues, labels=scenario_names)
        ax1.set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')
        ax1.set_title('Revenue Distribution by Scenario', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Mean Revenue by Scenario with Error Bars
        ax2 = axes[0, 1]
        means = [np.mean(self.all_results[scn]['total_revenue']) for scn in self.scenarios]
        stds = [np.std(self.all_results[scn]['total_revenue']) for scn in self.scenarios]
        
        x_pos = np.arange(len(scenario_names))
        colors_grad = plt.cm.viridis(np.linspace(0, 1, len(scenario_names)))
        ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors_grad, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Total Revenue ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Mean Revenue by Scenario (with Std Dev)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Investment Cost vs Mean Revenue
        ax3 = axes[1, 0]
        invest_costs = [SCENARIOS[scn]['invest_cost'] for scn in self.scenarios]
        ax3.scatter(invest_costs, means, s=150, c=means, cmap='viridis', edgecolors='black', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Investment Cost ($)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Mean Total Revenue ($)', fontsize=11, fontweight='bold')
        ax3.set_title('Investment vs Revenue', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(invest_costs, means, 1)
        p = np.poly1d(z)
        sorted_costs = sorted(invest_costs)
        ax3.plot(sorted_costs, p(sorted_costs), "r--", lw=2, alpha=0.8, label='Trend')
        ax3.legend()
        
        # Plot 4: Return on Investment (ROI)
        ax4 = axes[1, 1]
        base_revenue = np.mean(self.all_results['BASE']['total_revenue'])
        rois = [(means[i] - base_revenue) / max(invest_costs[i], 1) * 100 if invest_costs[i] > 0 else 0 
                for i in range(len(scenario_names))]
        
        colors_roi = ['green' if roi > 0 else 'red' for roi in rois]
        ax4.bar(x_pos, rois, color=colors_roi, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax4.set_ylabel('ROI (% per Dollar Invested)', fontsize=11, fontweight='bold')
        ax4.set_title('Return on Investment by Scenario', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', lw=1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Scenario comparison plot saved as 'scenario_comparison.png'")
        plt.show()


def main():
    """Main execution function."""
    print("SeaWorld Simulation - Multiple Linear Regression Analysis")
    print("="*80)
    print(f"\nModel: Total Revenue ~ avg_rating + avg_food_income + total_customers")
    print("="*80)
    
    # Initialize analyzer
    analyzer = LinearRegressionAnalyzer(alpha=ALPHA)
    
    # Run simulations
    analyzer.run_simulations(INITIAL_RUNS, base_seed=BASE_SEED)
    
    # Prepare data for regression
    analyzer.prepare_regression_data()
    
    # Perform regression analysis
    analyzer.analyze_regression()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate comprehensive visualizations
    print("\n" + "="*80)
    print("GENERATING ANALYTICAL PLOTS...")
    print("="*80)
    
    print("\n1. Generating main regression diagnostics plot...")
    analyzer.generate_analytics_plots()
    
    print("\n2. Generating coefficient significance plot...")
    analyzer.generate_coefficient_significance_plot()
    
    print("\n3. Generating correlation heatmap...")
    analyzer.generate_predictor_correlation_heatmap()
    
    print("\n4. Generating scenario comparison plots...")
    analyzer.generate_scenario_comparison_plot()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - ALL PLOTS SAVED")
    print("="*80)
    print("\nGenerated Files:")
    print("  • regression_analysis.png - Main diagnostics (6 plots)")
    print("  • coefficient_significance.png - p-value comparison")
    print("  • correlation_heatmap.png - Variable correlations")
    print("  • scenario_comparison.png - Revenue & ROI analysis")
    
    return analyzer


if __name__ == "__main__":
    import subprocess
    import sys

    print("Deprecated: use simulate.py and analyze.py for the new pipeline.")
    subprocess.run([sys.executable, "simulate.py"], check=True)
    subprocess.run([sys.executable, "analyze.py", "--in", "data/simulation_results.csv", "--no-plots"], check=True)
