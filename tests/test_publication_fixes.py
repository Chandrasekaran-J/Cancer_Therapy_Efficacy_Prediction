"""Small regression checks; no study data or trained models are required."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_ml"))
from feature_schema import reaction_activity, reaction_feature_row
from plot_shap_importance import classify_features, plot_shap_importance

spec = importlib.util.spec_from_file_location("scatter", ROOT / "src/06_visualization/biomarker_scatter.py")
scatter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scatter)


class PublicationFixTests(unittest.TestCase):
    def test_prediction_matches_training_for_signed_flux_and_prefixes(self):
        values = np.array([-2.0, 0, 1e-9, 2e-9])
        names = ["rxn_A", "rxn_B", "flux_C", "rxn_D"]
        actual = reaction_feature_row(values, names, dict(zip("ABCD", range(4))))
        self.assertEqual(list(actual.values()), [1, 0, 0, 1])
        np.testing.assert_array_equal(list(actual.values()), reaction_activity(values))

    def test_missing_required_reaction_is_not_silently_zero(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            reaction_feature_row([1], ["rxn_absent"], {"present": 0})

    def test_shap_reclassification_preserves_values_and_plots_both_types(self):
        source = pd.DataFrame({"feature": ["rxn_A", "age", "flux_B"],
                               "mean_abs_shap": [0.2, 0.5, 0.1],
                               "feature_type": ["clinical"] * 3})
        corrected = classify_features(source).set_index("feature")
        self.assertEqual(corrected.loc["rxn_A", "feature_type"], "metabolic_flux")
        self.assertEqual(corrected.loc["flux_B", "feature_type"], "metabolic_flux")
        self.assertEqual(corrected.loc["age", "feature_type"], "clinical")
        self.assertEqual(corrected.loc["rxn_A", "mean_abs_shap"], 0.2)
        with tempfile.TemporaryDirectory() as directory:
            with patch("matplotlib.axes.Axes.barh", autospec=True) as bars:
                plot_shap_importance(source, "Chemotherapy", directory)
            self.assertEqual(list(bars.call_args.kwargs["color"]),
                             ["darkorange", "steelblue", "steelblue"])
            for ext in ("png", "pdf"):
                self.assertGreater((Path(directory) / f"Chemotherapy_shap_bar.{ext}").stat().st_size, 0)

    def test_scatter_retains_shared_effects_and_labels_projections(self):
        chemo = pd.DataFrame({"rxn_id": ["shared", "chemo"],
                              "log2_fold_change": [-1.5, 0.6], "rxn_name": ["S", "C"]})
        radio = pd.DataFrame({"rxn_id": ["shared", "radio"],
                              "log2_fold_change": [-2, 1.2], "rxn_name": ["S", "R"]})
        fig = scatter.build_figure(chemo, radio)
        self.assertEqual(fig.data[0].x[0], scatter.to_display(-1.5, scatter.X_REAL, scatter.X_DISPLAY))
        self.assertIn("-1.500", fig.data[0].customdata[0])
        for trace in fig.data[1:]:
            self.assertIn("projected", trace.name)
            self.assertIn("axis projection", trace.customdata[0])
        self.assertIn("piecewise-linear", fig.layout.annotations[0].text)
        self.assertGreater(scatter.to_display(2, scatter.X_REAL, scatter.X_DISPLAY), scatter.X_DISPLAY[-1])


if __name__ == "__main__":
    unittest.main()
