import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops identifier or leakage columns inside the pipeline."""

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        cols = self.cols_to_drop if self.cols_to_drop is not None else ["person_id"]
        existing = [c for c in cols if c in X_out.columns]
        return X_out.drop(columns=existing)


class RangeCleaner(BaseEstimator, TransformerMixin):
    """Replaces physically impossible numeric values with np.nan for downstream imputation."""

    DEFAULT_RULES = {
        "age": (15, 100),
        "gpa_semester_1": (0.0, 4.0),
        "gpa_semester_2": (0.0, 4.0),
        "attendance_rate": (0.0, 100.0),
        "assignments_submitted_rate": (0.0, 100.0),
        "admission_grade": (0.0, 200.0),
        "previous_qualification_grade": (0.0, 200.0),
        "weekly_work_hours": (0.0, 80.0),
        "sleep_hours": (0.0, 24.0),
        "stress_level": (1, 10),
        "satisfaction_score": (1, 10),
    }

    def __init__(self, rules=None):
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        active_rules = self.rules if self.rules is not None else self.DEFAULT_RULES
        for col, (lo, hi) in active_rules.items():
            if col in X_out.columns:
                invalid = (X_out[col] < lo) | (X_out[col] > hi)
                X_out.loc[invalid, col] = np.nan
        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-engineered features; supports add_features=False for ablation."""

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if not self.add_features:
            return X_out

        # 1. Pass rate: captures credit completion efficiency
        if {"units_passed", "units_enrolled"}.issubset(X_out.columns):
            X_out["pass_rate"] = X_out["units_passed"] / X_out["units_enrolled"].replace(0, np.nan)

        # 2. GPA change: captures academic trajectory per semester
        if {"gpa_semester_2", "gpa_semester_1"}.issubset(X_out.columns):
            X_out["gpa_change"] = X_out["gpa_semester_2"] - X_out["gpa_semester_1"]

        # 3. Engagement index: combines attendance and assignment submission discipline
        if {"attendance_rate", "assignments_submitted_rate"}.issubset(X_out.columns):
            X_out["engagement_index"] = 0.5 * X_out["attendance_rate"] + 0.5 * X_out["assignments_submitted_rate"]

        # 4. Work hours per enrolled unit: measures employment time pressure per credit
        if {"weekly_work_hours", "units_enrolled"}.issubset(X_out.columns):
            X_out["work_hours_per_unit"] = X_out["weekly_work_hours"] / X_out["units_enrolled"].replace(0, np.nan)

        # 5. Financial risk flag: flags students with overdue tuition and no scholarship support
        if {"tuition_up_to_date", "scholarship_holder"}.issubset(X_out.columns):
            X_out["financial_risk_flag"] = (
                (X_out["tuition_up_to_date"] == 0) & (X_out["scholarship_holder"] == 0)
            ).astype(int)

        return X_out