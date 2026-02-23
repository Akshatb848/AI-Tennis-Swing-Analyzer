"""Exploratory Analysis Agent -- computes statistics, tests, and visualisation specs."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base import AgentResult, BaseAgent


class ExploratoryAnalysisAgent(BaseAgent):
    """Performs thorough automated exploratory data analysis.

    Includes normality testing, correlation significance, ANOVA/Kruskal-Wallis,
    chi-square independence, distribution metrics, multicollinearity (VIF),
    and auto-generated visualisation specifications.
    """

    @property
    def name(self) -> str:
        return "ExploratoryAnalysisAgent"

    def execute(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        try:
            self.logger.info(
                "ExploratoryAnalysisAgent started -- shape %s, target=%s",
                df.shape, target_col,
            )
            df = df.copy()

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            distribution_analysis = self._analyze_distributions(df, numeric_cols)
            normality_tests = self._test_normality(df, numeric_cols)
            corr_matrix = self._compute_correlation(df, numeric_cols)
            corr_significance = self._correlation_significance(df, numeric_cols)
            hypothesis_tests = self._run_hypothesis_tests(
                df, target_col, problem_type, numeric_cols, cat_cols,
            )
            vif_results = self._compute_vif(df, numeric_cols, target_col)
            viz_specs = self._generate_viz_specs(
                df, target_col, problem_type, numeric_cols, cat_cols, corr_matrix,
            )
            summary = self._build_summary(
                df, target_col, problem_type, numeric_cols, cat_cols,
                corr_matrix, hypothesis_tests, distribution_analysis, normality_tests, vif_results,
            )

            corr_dict = {}
            if corr_matrix is not None:
                corr_dict = {
                    col: {row: round(float(corr_matrix.loc[row, col]), 6) for row in corr_matrix.index}
                    for col in corr_matrix.columns
                }

            all_tests = normality_tests + corr_significance + hypothesis_tests
            all_tests.sort(key=lambda x: x.get("p_value") if x.get("p_value") is not None else 1.0)

            self.logger.info(
                "ExploratoryAnalysisAgent complete -- %d tests, %d viz specs",
                len(all_tests), len(viz_specs),
            )

            return AgentResult(
                success=True,
                data={
                    "correlation_matrix": corr_dict,
                    "statistical_tests": all_tests,
                    "hypothesis_tests": hypothesis_tests,
                    "normality_tests": normality_tests,
                    "correlation_significance": corr_significance,
                    "distribution_analysis": distribution_analysis,
                    "vif_results": vif_results,
                    "viz_specs": viz_specs,
                    "summary": summary,
                },
                metadata={"target_col": target_col, "problem_type": problem_type},
            )

        except Exception as exc:
            self.logger.error("ExploratoryAnalysisAgent failed: %s", exc, exc_info=True)
            return AgentResult(success=False, errors=[str(exc)])

    def _analyze_distributions(
        self, df: pd.DataFrame, numeric_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        for col in numeric_cols[:20]:
            s = df[col].dropna()
            if len(s) < 5:
                continue
            skewness = float(s.skew())
            kurtosis = float(s.kurtosis())

            if abs(skewness) < 0.5:
                skew_label = "Symmetric"
            elif skewness > 0:
                skew_label = "Right-skewed" if skewness < 1.5 else "Highly right-skewed"
            else:
                skew_label = "Left-skewed" if skewness > -1.5 else "Highly left-skewed"

            if kurtosis < -1:
                kurt_label = "Platykurtic (light tails)"
            elif kurtosis > 1:
                kurt_label = "Leptokurtic (heavy tails)"
            else:
                kurt_label = "Mesokurtic (normal-like)"

            results.append({
                "feature": col,
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "skew_label": skew_label,
                "kurtosis_label": kurt_label,
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "cv": round(float(s.std() / s.mean()), 4) if abs(s.mean()) > 1e-10 else None,
                "unique_ratio": round(float(s.nunique() / len(s)), 4),
            })
        return results

    def _test_normality(
        self, df: pd.DataFrame, numeric_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        try:
            from scipy import stats as sp_stats
        except ImportError:
            return results

        for col in numeric_cols[:15]:
            s = df[col].dropna()
            if len(s) < 8:
                continue
            try:
                if len(s) <= 5000:
                    stat, p_val = sp_stats.shapiro(s.values[:5000])
                    test_name = "Shapiro-Wilk"
                else:
                    stat, p_val = sp_stats.normaltest(s.values)
                    test_name = "D'Agostino-Pearson"

                is_normal = p_val > 0.05
                results.append({
                    "feature": col,
                    "test": test_name,
                    "category": "normality",
                    "statistic": round(float(stat), 6) if np.isfinite(stat) else None,
                    "p_value": round(float(p_val), 6) if np.isfinite(p_val) else None,
                    "conclusion": "Normal distribution" if is_normal else "Non-normal distribution",
                    "significant": not is_normal,
                })
            except Exception:
                continue
        return results

    def _compute_correlation(
        self, df: pd.DataFrame, numeric_cols: List[str],
    ) -> Optional[pd.DataFrame]:
        if len(numeric_cols) < 2:
            return None
        try:
            return df[numeric_cols].corr()
        except Exception:
            return None

    def _correlation_significance(
        self, df: pd.DataFrame, numeric_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        try:
            from scipy import stats as sp_stats
        except ImportError:
            return results

        cols = numeric_cols[:10]
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                try:
                    valid = df[[c1, c2]].dropna()
                    if len(valid) < 5:
                        continue
                    r_pearson, p_pearson = sp_stats.pearsonr(valid[c1], valid[c2])
                    r_spearman, p_spearman = sp_stats.spearmanr(valid[c1], valid[c2])
                    results.append({
                        "feature": f"{c1} vs {c2}",
                        "test": "Pearson & Spearman",
                        "category": "correlation",
                        "statistic": round(float(r_pearson), 6),
                        "p_value": round(float(p_pearson), 6),
                        "spearman_r": round(float(r_spearman), 6),
                        "spearman_p": round(float(p_spearman), 6),
                        "conclusion": self._interpret_correlation(r_pearson, p_pearson),
                        "significant": p_pearson < 0.05,
                    })
                except Exception:
                    continue
        return results

    def _interpret_correlation(self, r: float, p: float) -> str:
        if p >= 0.05:
            return "No significant correlation"
        strength = abs(r)
        if strength > 0.8:
            label = "Very strong"
        elif strength > 0.6:
            label = "Strong"
        elif strength > 0.4:
            label = "Moderate"
        elif strength > 0.2:
            label = "Weak"
        else:
            label = "Very weak"
        direction = "positive" if r > 0 else "negative"
        return f"{label} {direction} correlation (r={r:.3f})"

    def _run_hypothesis_tests(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        problem_type: Optional[str],
        numeric_cols: List[str],
        cat_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if target_col is None or target_col not in df.columns:
            return results

        is_target_categorical = (
            problem_type in ("binary_classification", "multiclass_classification")
            or df[target_col].nunique() <= 20
        )

        try:
            from scipy import stats as sp_stats
        except ImportError:
            return results

        if is_target_categorical:
            target_series = df[target_col].astype(str)
            unique_groups = target_series.unique()

            for col in numeric_cols:
                if col == target_col:
                    continue
                try:
                    groups = [
                        df.loc[target_series == g, col].dropna().values
                        for g in unique_groups
                    ]
                    groups = [g for g in groups if len(g) >= 2]
                    if len(groups) < 2:
                        continue

                    all_normal = True
                    for g in groups:
                        if len(g) >= 8:
                            _, p_norm = sp_stats.shapiro(g[:5000])
                            if p_norm < 0.05:
                                all_normal = False
                                break

                    homoscedastic = True
                    if len(groups) >= 2:
                        try:
                            _, p_lev = sp_stats.levene(*groups)
                            homoscedastic = p_lev > 0.05
                        except Exception:
                            pass

                    if all_normal and homoscedastic:
                        stat, p_val = sp_stats.f_oneway(*groups)
                        test_name = "One-way ANOVA"
                        assumption = "Parametric (normal + homoscedastic)"
                    else:
                        stat, p_val = sp_stats.kruskal(*groups)
                        test_name = "Kruskal-Wallis"
                        assumption = "Non-parametric"

                    if np.isfinite(stat) and np.isfinite(p_val):
                        n_total = sum(len(g) for g in groups)
                        k = len(groups)
                        if test_name == "One-way ANOVA" and p_val < 0.05:
                            eta_sq = (stat * (k - 1)) / (stat * (k - 1) + (n_total - k))
                        else:
                            eta_sq = None

                        results.append({
                            "feature": col,
                            "test": test_name,
                            "category": "group_comparison",
                            "statistic": round(float(stat), 6),
                            "p_value": round(float(p_val), 6),
                            "conclusion": f"Significant difference (p={p_val:.4f})" if p_val < 0.05
                                else f"No significant difference (p={p_val:.4f})",
                            "significant": p_val < 0.05,
                            "assumption": assumption,
                            "effect_size": round(float(eta_sq), 4) if eta_sq is not None else None,
                        })
                except Exception:
                    continue

            for col in cat_cols:
                if col == target_col:
                    continue
                try:
                    contingency = pd.crosstab(df[col], df[target_col])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    chi2, p_val, dof, expected = sp_stats.chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0

                    results.append({
                        "feature": col,
                        "test": "Chi-Square Independence",
                        "category": "independence",
                        "statistic": round(float(chi2), 6),
                        "p_value": round(float(p_val), 6),
                        "dof": int(dof),
                        "conclusion": f"Dependent (p={p_val:.4f}, V={cramers_v:.3f})" if p_val < 0.05
                            else f"Independent (p={p_val:.4f})",
                        "significant": p_val < 0.05,
                        "effect_size": round(float(cramers_v), 4),
                    })
                except Exception:
                    continue
        else:
            for col in numeric_cols:
                if col == target_col:
                    continue
                try:
                    valid = df[[col, target_col]].dropna()
                    if len(valid) < 5:
                        continue
                    r_val, p_val = sp_stats.pearsonr(valid[col], valid[target_col])
                    r_sp, p_sp = sp_stats.spearmanr(valid[col], valid[target_col])

                    results.append({
                        "feature": col,
                        "test": "Pearson Correlation",
                        "category": "correlation_with_target",
                        "statistic": round(float(r_val), 6),
                        "p_value": round(float(p_val), 6),
                        "spearman_r": round(float(r_sp), 6),
                        "conclusion": self._interpret_correlation(r_val, p_val),
                        "significant": p_val < 0.05,
                    })
                except Exception:
                    continue

        results.sort(key=lambda x: x.get("p_value") if x.get("p_value") is not None else 1.0)
        return results

    def _compute_vif(
        self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str],
    ) -> List[Dict[str, Any]]:
        results = []
        cols = [c for c in numeric_cols if c != target_col][:15]
        if len(cols) < 2:
            return results

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = df[cols].dropna()
            if len(X) < len(cols) + 1:
                return results
            X_array = X.values.astype(float)
            X_array = np.column_stack([np.ones(len(X_array)), X_array])

            for i, col in enumerate(cols):
                try:
                    vif_val = variance_inflation_factor(X_array, i + 1)
                    if np.isfinite(vif_val):
                        if vif_val > 10:
                            concern = "High multicollinearity"
                        elif vif_val > 5:
                            concern = "Moderate multicollinearity"
                        else:
                            concern = "Acceptable"
                        results.append({
                            "feature": col,
                            "vif": round(float(vif_val), 4),
                            "concern": concern,
                        })
                except Exception:
                    continue
        except ImportError:
            try:
                from numpy.linalg import inv
                X = df[cols].dropna()
                if len(X) < len(cols) + 1:
                    return results
                corr_matrix = X.corr().values
                try:
                    inv_corr = inv(corr_matrix)
                    for i, col in enumerate(cols):
                        vif_val = float(inv_corr[i, i])
                        if np.isfinite(vif_val):
                            if vif_val > 10:
                                concern = "High multicollinearity"
                            elif vif_val > 5:
                                concern = "Moderate multicollinearity"
                            else:
                                concern = "Acceptable"
                            results.append({
                                "feature": col,
                                "vif": round(vif_val, 4),
                                "concern": concern,
                            })
                except Exception:
                    pass
            except Exception:
                pass

        return results

    def _generate_viz_specs(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        problem_type: Optional[str],
        numeric_cols: List[str],
        cat_cols: List[str],
        corr_matrix: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []

        if target_col and target_col in df.columns:
            is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
            specs.append({
                "type": "histogram" if is_numeric_target else "bar",
                "title": f"Target Distribution: {target_col}",
                "x": target_col,
                "plot_kind": "target_distribution",
            })

        for col in numeric_cols[:6]:
            if col == target_col:
                continue
            specs.append({
                "type": "histogram",
                "title": f"Distribution: {col}",
                "x": col,
                "plot_kind": "feature_distribution",
            })

        for col in cat_cols[:4]:
            if col == target_col:
                continue
            specs.append({
                "type": "bar",
                "title": f"Distribution: {col}",
                "x": col,
                "plot_kind": "feature_distribution",
            })

        if corr_matrix is not None and corr_matrix.shape[0] >= 2:
            specs.append({
                "type": "heatmap",
                "title": "Correlation Heatmap",
                "data_key": "correlation_matrix",
                "plot_kind": "correlation_heatmap",
            })

            if target_col and target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col, errors="ignore").abs().sort_values(ascending=False)
                top_corr_features = target_corr.head(4).index.tolist()
                for feat in top_corr_features:
                    specs.append({
                        "type": "scatter",
                        "title": f"{feat} vs {target_col}",
                        "x": feat,
                        "y": target_col,
                        "plot_kind": "scatter_correlation",
                    })

        if target_col and target_col in df.columns:
            is_target_cat = (
                problem_type in ("binary_classification", "multiclass_classification")
                or df[target_col].nunique() <= 20
            )
            if not is_target_cat:
                for col in cat_cols[:3]:
                    if col == target_col:
                        continue
                    specs.append({
                        "type": "box",
                        "title": f"{target_col} by {col}",
                        "x": col,
                        "y": target_col,
                        "plot_kind": "box_categorical",
                    })
            else:
                for col in numeric_cols[:4]:
                    if col == target_col:
                        continue
                    specs.append({
                        "type": "box",
                        "title": f"{col} by {target_col}",
                        "x": target_col,
                        "y": col,
                        "plot_kind": "box_categorical",
                    })

        return specs

    def _build_summary(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        problem_type: Optional[str],
        numeric_cols: List[str],
        cat_cols: List[str],
        corr_matrix: Optional[pd.DataFrame],
        hypothesis_tests: List[Dict[str, Any]],
        distribution_analysis: List[Dict[str, Any]],
        normality_tests: List[Dict[str, Any]],
        vif_results: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = ["Exploratory Data Analysis Summary", ""]

        lines.append(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        lines.append(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(cat_cols)}")
        if target_col:
            lines.append(f"Target: {target_col} | Problem: {problem_type or 'auto-detected'}")
        lines.append("")

        total_missing = int(df.isnull().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        lines.append(f"Data Quality: {total_missing} missing values ({missing_pct:.1f}%)")
        lines.append("")

        skewed_features = [d for d in distribution_analysis if abs(d.get("skewness", 0)) > 1]
        if skewed_features:
            lines.append(f"Distribution: {len(skewed_features)} highly skewed features detected")
            for d in skewed_features[:3]:
                lines.append(f"  {d['feature']}: skew={d['skewness']}, {d['skew_label']}")
            lines.append("")

        normal_count = sum(1 for t in normality_tests if not t.get("significant", True))
        non_normal = sum(1 for t in normality_tests if t.get("significant", False))
        if normality_tests:
            lines.append(f"Normality: {normal_count} normal, {non_normal} non-normal (of {len(normality_tests)} tested)")
            lines.append("")

        if corr_matrix is not None and target_col and target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].drop(target_col, errors="ignore").abs().sort_values(ascending=False)
            lines.append(f"Top correlations with {target_col}:")
            for feat, val in target_corr.head(5).items():
                lines.append(f"  {feat}: r={val:.4f}")
            lines.append("")

        sig_tests = [t for t in hypothesis_tests if t.get("significant", False)]
        if hypothesis_tests:
            lines.append(f"Hypothesis Tests: {len(sig_tests)} significant of {len(hypothesis_tests)} total (alpha=0.05)")
            for t in sig_tests[:5]:
                lines.append(f"  {t['feature']} ({t['test']}): p={t.get('p_value', 'N/A')}")
            lines.append("")

        high_vif = [v for v in vif_results if v.get("vif", 0) > 5]
        if high_vif:
            lines.append(f"Multicollinearity: {len(high_vif)} features with VIF > 5")
            for v in high_vif[:5]:
                lines.append(f"  {v['feature']}: VIF={v['vif']} ({v['concern']})")
            lines.append("")

        return "\n".join(lines)
