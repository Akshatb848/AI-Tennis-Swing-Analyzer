"""Data Engineering Agent -- cleans, transforms, and prepares data for modelling."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from agents.base import AgentResult, BaseAgent


class DataEngineeringAgent(BaseAgent):
    """Production-grade data engineering pipeline.

    Analyses the nature of each feature before choosing transformations.
    Pipeline: drop columns -> impute -> outlier handling -> skewness correction ->
    encode categoricals -> feature interactions -> scale -> PCA -> MI selection -> split.
    Produces a comprehensive before/after report.
    """

    @property
    def name(self) -> str:
        return "DataEngineeringAgent"

    def execute(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: str = "regression",
        id_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        try:
            self.logger.info(
                "DataEngineeringAgent started -- shape %s, target=%s, problem=%s",
                df.shape, target_col, problem_type,
            )
            df = df.copy()
            report: Dict[str, Any] = {}

            before_stats = self._snapshot_stats(df, target_col)
            report["before_stats"] = before_stats

            leakage_columns: List[str] = kwargs.get("leakage_columns", [])
            drop_cols = list(set((id_columns or []) + leakage_columns))
            drop_cols = [c for c in drop_cols if c in df.columns and c != target_col]
            if drop_cols:
                df = df.drop(columns=drop_cols)
                self.logger.info("Dropped ID/leakage columns: %s", drop_cols)
            report["dropped_id_leakage"] = drop_cols

            high_missing_threshold = kwargs.get("high_missing_threshold", 0.6)
            missing_ratio = df.isnull().mean()
            high_missing = missing_ratio[missing_ratio > high_missing_threshold].index.tolist()
            high_missing = [c for c in high_missing if c != target_col]
            if high_missing:
                df = df.drop(columns=high_missing)
                self.logger.info("Dropped high-missing columns (>%d%%): %s",
                                 int(high_missing_threshold * 100), high_missing)
            report["dropped_high_missing"] = high_missing

            df, impute_info = self._impute_missing(df, target_col)
            report["imputation"] = impute_info

            df, outlier_info = self._handle_outliers(df, target_col)
            report["outlier_handling"] = outlier_info

            if target_col and target_col in df.columns:
                y = df[target_col].copy()
                X = df.drop(columns=[target_col])
            else:
                y = pd.Series(dtype=float)
                X = df

            if target_col and not pd.api.types.is_numeric_dtype(y) and len(y) > 0:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=target_col)
                report["target_label_encoded"] = True
                report["target_classes"] = list(le.classes_)
            else:
                report["target_label_encoded"] = False

            X, skew_info = self._correct_skewness(X)
            report["skewness_correction"] = skew_info

            X, enc_info = self._encode_categoricals(X, y, problem_type)
            report["encoding"] = enc_info

            remaining_obj = X.select_dtypes(include=["object", "category"]).columns.tolist()
            if remaining_obj:
                X = X.drop(columns=remaining_obj)
                report["dropped_remaining_non_numeric"] = remaining_obj

            X, interact_info = self._create_interactions(X, y, problem_type)
            report["interactions"] = interact_info

            X, scale_info = self._scale_features(X)
            report["scaling"] = scale_info

            X, pca_info = self._apply_pca(X)
            report["pca"] = pca_info

            X, mi_info = self._mutual_information_selection(X, y, problem_type)
            report["feature_selection"] = mi_info

            feature_names = X.columns.tolist()

            after_stats = self._snapshot_stats(
                pd.concat([X, y.rename(target_col)], axis=1) if target_col and len(y) > 0 else X,
                target_col,
            )
            report["after_stats"] = after_stats
            report["transformations_summary"] = self._build_transformations_summary(report)

            if target_col and len(y) > 0:
                stratify = y if "classification" in problem_type else None
                if stratify is not None:
                    min_class = int(y.value_counts().min())
                    if min_class < 2:
                        stratify = None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify,
                )
            else:
                X_train, X_test = X, pd.DataFrame(columns=X.columns)
                y_train, y_test = y, pd.Series(dtype=float)

            self.logger.info(
                "DataEngineeringAgent complete -- features=%d, train=%d, test=%d",
                len(feature_names), len(X_train), len(X_test),
            )

            return AgentResult(
                success=True,
                data={
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "feature_names": feature_names,
                    "engineering_report": report,
                },
                metadata={"problem_type": problem_type, "target_col": target_col},
            )

        except Exception as exc:
            self.logger.error("DataEngineeringAgent failed: %s", exc, exc_info=True)
            return AgentResult(success=False, errors=[str(exc)])

    def _snapshot_stats(self, df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        stats: Dict[str, Any] = {
            "shape": list(df.shape),
            "numeric_count": len(numeric_cols),
            "categorical_count": len(cat_cols),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_pct": round(float(df.isnull().mean().mean() * 100), 2),
        }
        feature_stats = {}
        for col in numeric_cols[:20]:
            s = df[col].dropna()
            if len(s) < 3:
                continue
            feature_stats[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "missing_pct": round(float(df[col].isnull().mean() * 100), 2),
            }
        stats["features"] = feature_stats
        return stats

    def _impute_missing(
        self, df: pd.DataFrame, target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"method": "adaptive", "columns_imputed": [], "details": {}}

        cols_with_missing = [c for c in df.columns if df[c].isnull().any()]
        if not cols_with_missing:
            return df, info

        numeric_missing = [c for c in cols_with_missing if pd.api.types.is_numeric_dtype(df[c])]
        cat_missing = [c for c in cols_with_missing if c not in numeric_missing]

        for col in numeric_missing:
            missing_pct = df[col].isnull().mean()
            s = df[col].dropna()
            if len(s) < 3:
                df[col] = df[col].fillna(0)
                info["details"][col] = {"method": "zero_fill", "reason": "too few values"}
                continue
            skew = abs(s.skew())
            if missing_pct < 0.05:
                if skew > 1:
                    fill_val = s.median()
                    method = "median"
                else:
                    fill_val = s.mean()
                    method = "mean"
                df[col] = df[col].fillna(fill_val)
                info["details"][col] = {"method": method, "fill_value": round(float(fill_val), 4)}
            else:
                try:
                    from sklearn.impute import KNNImputer
                    num_subset = df[[col]].copy()
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(num_subset).ravel()
                    info["details"][col] = {"method": "knn", "missing_pct": round(float(missing_pct * 100), 1)}
                except Exception:
                    df[col] = df[col].fillna(s.median())
                    info["details"][col] = {"method": "median_fallback"}

        for col in cat_missing:
            missing_pct = df[col].isnull().mean()
            mode_vals = df[col].mode()
            if missing_pct > 0.1:
                df[col] = df[col].fillna("_MISSING_")
                info["details"][col] = {"method": "missing_indicator"}
            elif len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
                info["details"][col] = {"method": "mode", "fill_value": str(mode_vals.iloc[0])}
            else:
                df[col] = df[col].fillna("UNKNOWN")
                info["details"][col] = {"method": "unknown"}

        info["columns_imputed"] = cols_with_missing
        info["numeric_imputed"] = numeric_missing
        info["categorical_imputed"] = cat_missing
        return df, info

    def _handle_outliers(
        self, df: pd.DataFrame, target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"columns_handled": [], "details": {}}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outlier_count = int(outlier_mask.sum())
            outlier_pct = outlier_count / len(df) * 100

            if outlier_count == 0:
                continue

            if outlier_pct > 10:
                method = "winsorize"
                df[col] = df[col].clip(lower=s.quantile(0.05), upper=s.quantile(0.95))
            else:
                method = "iqr_clip"
                df[col] = df[col].clip(lower=lower, upper=upper)

            info["columns_handled"].append(col)
            info["details"][col] = {
                "method": method,
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_pct, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }

        self.logger.info("Outlier handling applied to %d columns", len(info["columns_handled"]))
        return df, info

    def _correct_skewness(
        self, X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"corrections": {}, "skipped": []}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            s = X[col].dropna()
            if len(s) < 10:
                continue
            skew = float(s.skew())
            if abs(skew) < 0.75:
                continue

            min_val = float(s.min())
            if min_val > 0:
                try:
                    from scipy.stats import boxcox
                    transformed, lam = boxcox(s.values + 1e-10)
                    X[col] = pd.Series(transformed, index=s.index)
                    X[col] = X[col].reindex(X.index)
                    new_skew = float(X[col].dropna().skew())
                    info["corrections"][col] = {
                        "method": "box_cox",
                        "original_skewness": round(skew, 4),
                        "new_skewness": round(new_skew, 4),
                        "lambda": round(float(lam), 4),
                    }
                    self.logger.info("Box-Cox applied to %s: skew %.3f -> %.3f", col, skew, new_skew)
                    continue
                except Exception:
                    pass

            try:
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                vals = X[col].values.reshape(-1, 1)
                mask = ~np.isnan(vals.ravel())
                if mask.sum() < 10:
                    info["skipped"].append(col)
                    continue
                result = np.full_like(vals, np.nan)
                result[mask] = pt.fit_transform(vals[mask].reshape(-1, 1))
                X[col] = result.ravel()
                new_skew = float(X[col].dropna().skew())
                info["corrections"][col] = {
                    "method": "yeo_johnson",
                    "original_skewness": round(skew, 4),
                    "new_skewness": round(new_skew, 4),
                }
                self.logger.info("Yeo-Johnson applied to %s: skew %.3f -> %.3f", col, skew, new_skew)
            except Exception:
                if min_val >= 0:
                    X[col] = np.log1p(X[col])
                    new_skew = float(X[col].dropna().skew())
                    info["corrections"][col] = {
                        "method": "log1p",
                        "original_skewness": round(skew, 4),
                        "new_skewness": round(new_skew, 4),
                    }
                else:
                    info["skipped"].append(col)

        self.logger.info("Skewness correction applied to %d columns", len(info["corrections"]))
        return X, info

    def _encode_categoricals(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"encoded": {}, "dropped_high_cardinality": []}
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            nunique = X[col].nunique()
            if nunique <= 2:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                info["encoded"][col] = {
                    "method": "label",
                    "categories": nunique,
                    "mapping": dict(zip(le.classes_.tolist(), range(len(le.classes_)))),
                }
            elif nunique <= 15:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True).astype(int)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                info["encoded"][col] = {
                    "method": "onehot",
                    "categories": nunique,
                    "new_columns": dummies.columns.tolist(),
                }
            elif nunique <= 100 and len(y) > 0:
                try:
                    global_mean = float(y.mean())
                    means = y.groupby(X[col].astype(str)).mean()
                    smoothing = 10
                    counts = y.groupby(X[col].astype(str)).count()
                    smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
                    X[col] = X[col].astype(str).map(smooth_means).fillna(global_mean)
                    info["encoded"][col] = {
                        "method": "target_encoding",
                        "categories": nunique,
                        "smoothing": smoothing,
                    }
                except Exception:
                    X = X.drop(columns=[col])
                    info["dropped_high_cardinality"].append(col)
            else:
                X = X.drop(columns=[col])
                info["dropped_high_cardinality"].append(col)

        self.logger.info(
            "Encoded %d categorical columns, dropped %d high-cardinality",
            len(info["encoded"]), len(info["dropped_high_cardinality"]),
        )
        return X, info

    def _create_interactions(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"created": [], "method": "none"}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2 or len(numeric_cols) > 20 or len(y) == 0:
            return X, info

        if X.shape[1] > 30:
            return X, info

        try:
            if "classification" in problem_type:
                from sklearn.feature_selection import mutual_info_classif as mi_func
            else:
                from sklearn.feature_selection import mutual_info_regression as mi_func

            mi_scores = mi_func(X[numeric_cols].fillna(0), y, random_state=42)
            mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
            top_features = mi_series.head(min(5, len(mi_series))).index.tolist()

            created_features = []
            for i, f1 in enumerate(top_features):
                for f2 in top_features[i + 1:]:
                    if len(created_features) >= 5:
                        break
                    interact_name = f"{f1}_x_{f2}"
                    X[interact_name] = X[f1] * X[f2]
                    created_features.append(interact_name)

            if created_features:
                info["created"] = created_features
                info["method"] = "top_mi_interactions"
                self.logger.info("Created %d interaction features", len(created_features))
        except Exception as exc:
            self.logger.warning("Interaction creation failed: %s", exc)
            info["error"] = str(exc)

        return X, info

    def _scale_features(
        self, X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return X, {"method": "none", "columns": []}

        has_heavy_tails = False
        for col in numeric_cols[:10]:
            s = X[col].dropna()
            if len(s) > 10 and abs(float(s.kurtosis())) > 5:
                has_heavy_tails = True
                break

        if has_heavy_tails:
            scaler = RobustScaler()
            method = "robust"
        else:
            scaler = StandardScaler()
            method = "standard"

        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.logger.info("Applied %s scaling to %d columns", method, len(numeric_cols))
        return X, {"method": method, "columns": numeric_cols}

    def _apply_pca(
        self, X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"applied": False}
        if X.shape[1] <= 50:
            return X, info

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95, random_state=42)
            original_cols = X.shape[1]
            X_pca = pca.fit_transform(X)
            n_components = X_pca.shape[1]
            col_names = [f"PC_{i + 1}" for i in range(n_components)]
            X = pd.DataFrame(X_pca, columns=col_names, index=X.index)
            info.update({
                "applied": True,
                "original_features": original_cols,
                "n_components": n_components,
                "explained_variance": round(float(sum(pca.explained_variance_ratio_)), 4),
            })
            self.logger.info(
                "PCA reduced features from %d to %d (%.1f%% variance)",
                original_cols, n_components, info["explained_variance"] * 100,
            )
        except Exception as exc:
            self.logger.warning("PCA failed: %s", exc)
            info["error"] = str(exc)

        return X, info

    def _mutual_information_selection(
        self, X: pd.DataFrame, y: pd.Series, problem_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        info: Dict[str, Any] = {"applied": False}
        if len(y) == 0 or X.shape[1] <= 5:
            return X, info

        try:
            if "classification" in problem_type:
                from sklearn.feature_selection import mutual_info_classif as mi_func
            else:
                from sklearn.feature_selection import mutual_info_regression as mi_func

            X_clean = X.fillna(0)
            mi_scores = mi_func(X_clean, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

            zero_mi = mi_series[mi_series < 1e-6].index.tolist()
            if zero_mi and len(mi_series) - len(zero_mi) >= 2:
                X = X.drop(columns=zero_mi)
                info.update({
                    "applied": True,
                    "features_before": len(mi_series),
                    "features_after": X.shape[1],
                    "dropped_zero_mi": zero_mi,
                    "top_scores": {f: round(float(s), 6) for f, s in mi_series.head(10).items()},
                })
                self.logger.info("MI selection: dropped %d zero-MI features", len(zero_mi))
            else:
                info["applied"] = False
                info["reason"] = "all features retained"
                info["top_scores"] = {f: round(float(s), 6) for f, s in mi_series.head(10).items()}

        except Exception as exc:
            self.logger.warning("MI selection failed: %s", exc)
            info["error"] = str(exc)

        return X, info

    def _build_transformations_summary(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        summary = []

        before = report.get("before_stats", {})
        after = report.get("after_stats", {})
        summary.append({
            "step": "Dataset Shape",
            "before": f"{before.get('shape', ['?', '?'])[0]} rows x {before.get('shape', ['?', '?'])[1]} cols",
            "after": f"{after.get('shape', ['?', '?'])[0]} rows x {after.get('shape', ['?', '?'])[1]} cols",
        })

        summary.append({
            "step": "Missing Values",
            "before": f"{before.get('missing_pct', '?')}%",
            "after": f"{after.get('missing_pct', '?')}%",
        })

        dropped = report.get("dropped_id_leakage", []) + report.get("dropped_high_missing", [])
        if dropped:
            summary.append({
                "step": "Columns Dropped",
                "before": f"{len(dropped)} identified",
                "after": ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else ""),
            })

        impute = report.get("imputation", {})
        imputed_cols = impute.get("columns_imputed", [])
        if imputed_cols:
            methods = set()
            for col, detail in impute.get("details", {}).items():
                methods.add(detail.get("method", "unknown"))
            summary.append({
                "step": "Imputation",
                "before": f"{len(imputed_cols)} columns with missing",
                "after": f"Methods: {', '.join(methods)}",
            })

        outliers = report.get("outlier_handling", {})
        handled = outliers.get("columns_handled", [])
        if handled:
            summary.append({
                "step": "Outlier Handling",
                "before": f"{len(handled)} columns with outliers",
                "after": "IQR clip / winsorize applied",
            })

        skew = report.get("skewness_correction", {})
        corrections = skew.get("corrections", {})
        if corrections:
            methods = set()
            for col, detail in corrections.items():
                methods.add(detail.get("method", "unknown"))
            summary.append({
                "step": "Skewness Correction",
                "before": f"{len(corrections)} skewed features",
                "after": f"Methods: {', '.join(methods)}",
            })

        enc = report.get("encoding", {})
        encoded = enc.get("encoded", {})
        if encoded:
            methods = set()
            for col, detail in encoded.items():
                methods.add(detail.get("method", "unknown"))
            summary.append({
                "step": "Categorical Encoding",
                "before": f"{len(encoded)} categorical columns",
                "after": f"Methods: {', '.join(methods)}",
            })

        scale = report.get("scaling", {})
        if scale.get("method", "none") != "none":
            summary.append({
                "step": "Feature Scaling",
                "before": f"{len(scale.get('columns', []))} numeric columns",
                "after": f"{scale['method'].title()} scaler",
            })

        interactions = report.get("interactions", {})
        if interactions.get("created"):
            summary.append({
                "step": "Feature Interactions",
                "before": "N/A",
                "after": f"{len(interactions['created'])} interaction features created",
            })

        return summary
