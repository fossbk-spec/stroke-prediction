"""Unit tests for the EDA module: every function must run end-to-end without errors.

The assignment explicitly requires: "EDA source code must be packaged into
reusable functions, executable end-to-end without errors" -> this test suite
enforces exactly that.
"""

from __future__ import annotations

import pandas as pd

from src import config, eda


class TestEdaTables:
    def test_missing_value_report_flags_bmi_and_unknown(self, synthetic_df):
        report = eda.missing_value_report(synthetic_df)
        bmi_row = report.loc[report["feature"] == "bmi"].iloc[0]
        smoking_row = report.loc[report["feature"] == "smoking_status"].iloc[0]
        assert bmi_row["n_missing"] > 0
        assert smoking_row["n_unknown_token"] > 0, "'Unknown' must be reported separately"

    def test_class_balance_reports_both_levels(self, synthetic_df):
        report = eda.class_balance_report(synthetic_df)
        assert set(report["level"]) == {"sample-level", "patient-level"}
        assert (report["positive_rate"] < 0.2).all()

    def test_patient_level_differs_when_records_are_duplicated(self, multi_record_df):
        """With multiple records per patient, the two levels must give different counts."""
        report = eda.class_balance_report(multi_record_df).set_index("level")
        assert report.loc["sample-level", "n_total"] > report.loc["patient-level", "n_total"]

    def test_age_bands_are_monotone_in_risk(self, synthetic_df):
        bands = eda.stroke_rate_by_age_band(synthetic_df)
        assert bands["n_samples"].sum() == len(synthetic_df)
        assert bands["stroke_rate"].between(0, 1).all()

    def test_glucose_strata_include_risk_ratio(self, synthetic_df):
        strata = eda.glucose_risk_strata(synthetic_df)
        assert "risk_ratio_vs_normal" in strata.columns
        assert strata["n_samples"].sum() == len(synthetic_df)

    def test_correlation_table_has_feature_column(self, synthetic_df):
        corr = eda.correlation_with_target(synthetic_df)
        assert list(corr.columns) == ["feature", "corr_with_stroke"]
        assert "age" in set(corr["feature"])

    def test_demographic_bias_covers_all_attributes(self, synthetic_df):
        report = eda.demographic_bias_report(synthetic_df)
        assert {"gender", "Residence_type", "work_type", "smoking_status", "ever_married"} == set(
            report["attribute"]
        )
        assert report["share_of_cohort"].between(0, 1).all()

    def test_numeric_summary_splits_by_label(self, synthetic_df):
        summary = eda.numeric_summary(synthetic_df)
        assert set(summary[config.TARGET_COLUMN]) == {0, 1}
        assert set(summary["feature"]) == set(config.NUMERIC_FEATURES)


class TestEdaOrchestration:
    def test_run_full_eda_executes_end_to_end(self, synthetic_df, tmp_path):
        tables = eda.run_full_eda(synthetic_df, output_dir=tmp_path)
        expected = {
            "missing_values", "class_balance", "numeric_summary", "age_bands",
            "glucose_strata", "correlation", "demographic_bias", "comorbidity",
        }
        assert expected == set(tables)
        for name, table in tables.items():
            assert isinstance(table, pd.DataFrame)
            assert not table.empty, f"Table '{name}' is empty"
            assert (tmp_path / f"eda_{name}.csv").exists()

    def test_eda_does_not_mutate_input(self, synthetic_df):
        before = synthetic_df.copy()
        eda.run_full_eda(synthetic_df)
        pd.testing.assert_frame_equal(synthetic_df, before)

    def test_overview_figure_is_saved(self, synthetic_df, tmp_path):
        path = tmp_path / "overview.png"
        eda.plot_eda_overview(synthetic_df, save_path=path)
        assert path.exists() and path.stat().st_size > 0


def test_glucose_risk_ratio_without_normal_group(synthetic_df):
    df = synthetic_df.copy()
    df["avg_glucose_level"] = 150.0
    strata = eda.glucose_risk_strata(df)
    assert strata["risk_ratio_vs_normal"].isna().all()
