import unittest

from trustmargin_exp.trustmargin import arbitrate


class TrustMarginTests(unittest.TestCase):
    def test_formula(self):
        scores = {
            "q_only": {"d2l": -2.0, "rag": -1.0},
            "q_context": {"d2l": -1.2, "rag": -0.5},
            "context_only": {"d2l": -2.0, "rag": -2.0},
        }
        result = arbitrate(scores, lambda_bind=0.5, tau=0)
        self.assertAlmostEqual(result["m_prior"], 1)
        self.assertAlmostEqual(result["delta_d2l"], 0.8)
        self.assertAlmostEqual(result["delta_rag"], 1.5)
        self.assertAlmostEqual(result["m_bind"], 0.7)
        self.assertAlmostEqual(result["trust_score"], 1.35)
        self.assertEqual(result["selected_source"], "rag")

    def test_threshold_tie_goes_to_d2l(self):
        scores = {name: {"d2l": -1.0, "rag": -1.0} for name in ("q_only", "q_context", "context_only")}
        self.assertEqual(arbitrate(scores, tau=0)["selected_source"], "d2l")
        self.assertEqual(arbitrate(scores, tau=-0.1)["selected_source"], "rag")


if __name__ == "__main__":
    unittest.main()
