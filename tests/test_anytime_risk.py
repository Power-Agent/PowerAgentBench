import unittest
from unittest.mock import patch

from poweragentbench.steady_state_agentic import (
    AgentOutput,
    PFResult,
    SteadyN2ToolServer,
    compute_anytime_risk_metrics,
    score_agent,
)


class TestAnytimeRiskMetrics(unittest.TestCase):
    def setUp(self):
        self.A, self.B, self.C, self.D = (0, 1), (0, 2), (0, 3), (0, 4)
        self.oracle = {
            self.A: 4.0,
            self.B: 3.0,
            self.C: 2.0,
            self.D: 1.0,
        }
        self.dangerous = set(self.oracle)

    def output(self, order, budget=4, values=None, post_validated=None):
        values = self.oracle if values is None else values
        return AgentOutput(
            name="toy",
            validated={c: values[c] for c in order},
            reported=[],
            post_validated={} if post_validated is None else post_validated,
            validation_budget=float(budget),
        )

    def metrics(self, order, budget=4, oracle=None, dangerous=None, values=None, post_validated=None):
        return compute_anytime_risk_metrics(
            self.output(order, budget, values=values, post_validated=post_validated),
            self.oracle if oracle is None else oracle,
            self.dangerous if dangerous is None else dangerous,
        )

    def test_known_early_trajectory(self):
        result = self.metrics([self.A, self.B, self.C, self.D])

        self.assertAlmostEqual(result["anytime_risk_auc"], 0.625)
        self.assertAlmostEqual(result["anytime_risk_at_25"], 0.40)
        self.assertAlmostEqual(result["anytime_risk_at_50"], 0.70)
        self.assertAlmostEqual(result["anytime_risk_at_75"], 0.90)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 1.00)

    def test_fractional_checkpoints_use_interpolation(self):
        result = self.metrics([self.A, self.B, self.C, self.D], budget=10)

        # The discovery curve reaches 0.4 at validation 1 and 0.7 at 2.
        # Exact 25% and 75% budget positions are therefore interpolated.
        self.assertAlmostEqual(result["anytime_risk_at_25"], 0.80)
        self.assertAlmostEqual(result["anytime_risk_at_50"], 1.00)
        self.assertAlmostEqual(result["anytime_risk_at_75"], 1.00)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 1.00)

    def test_small_fractional_budget_positions(self):
        result = self.metrics([self.A, self.B], budget=2)
        self.assertAlmostEqual(result["anytime_risk_at_25"], 0.20)
        self.assertAlmostEqual(result["anytime_risk_at_50"], 0.40)
        self.assertAlmostEqual(result["anytime_risk_at_75"], 0.55)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 0.70)

    def test_early_discovery_beats_late_discovery(self):
        early = self.metrics([self.A, self.B, self.C, self.D])
        late = self.metrics([self.D, self.C, self.B, self.A])

        self.assertAlmostEqual(early["anytime_risk_auc"], 0.625)
        self.assertAlmostEqual(late["anytime_risk_auc"], 0.375)
        self.assertEqual(early["anytime_risk_at_100"], 1.0)
        self.assertEqual(late["anytime_risk_at_100"], 1.0)
        self.assertGreater(early["anytime_risk_auc"], late["anytime_risk_auc"])

    def test_validation_budget_limits_trajectory(self):
        result = self.metrics([self.A, self.B, self.C, self.D], budget=2)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 0.70)

    def test_empty_trajectory_with_positive_budget(self):
        result = self.metrics([], budget=4)
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_partially_consumed_budget_holds_final_discovery(self):
        result = self.metrics([self.A], budget=4)
        self.assertAlmostEqual(result["anytime_risk_auc"], 0.35)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 0.40)

    def test_mixed_dangerous_and_nondangerous_validations(self):
        dangerous = {self.A, self.C}
        result = self.metrics([self.B, self.A, self.D, self.C], budget=4, dangerous=dangerous)
        self.assertAlmostEqual(result["anytime_risk_at_25"], 0.0)
        self.assertAlmostEqual(result["anytime_risk_at_50"], 4.0 / 6.0)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 1.0)

    def test_hidden_oracle_weights_are_used(self):
        public_values = {self.A: 100.0, self.B: 1.0, self.C: 1.0, self.D: 1.0}
        hidden_oracle = {self.A: 4.0, self.B: 3.0, self.C: 2.0, self.D: 1.0}
        result = self.metrics(
            [self.B, self.A, self.C, self.D],
            oracle=hidden_oracle,
            values=public_values,
        )
        self.assertAlmostEqual(result["anytime_risk_at_50"], 0.70)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 1.0)

    def test_post_action_validations_do_not_enter_trajectory(self):
        result = self.metrics([self.A], budget=4, post_validated={self.B: 999.0})
        self.assertAlmostEqual(result["anytime_risk_at_100"], 0.40)

    def test_empty_or_zero_risk_cases_return_zero(self):
        no_validation = self.metrics([], budget=0)
        no_danger = self.metrics([self.A], dangerous=set())

        zero_oracle = {self.A: 0.0, self.B: 0.0}
        zero_risk = self.metrics(
            [self.A, self.B],
            budget=2,
            oracle=zero_oracle,
            dangerous={self.A, self.B},
        )

        for result in (no_validation, no_danger, zero_risk):
            self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_metrics_are_bounded(self):
        result = self.metrics([self.B, self.A, self.D, self.C])

        for name, value in result.items():
            with self.subTest(metric=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestScoreAgentAnytimeRiskIntegration(unittest.TestCase):
    def test_score_agent_exposes_anytime_risk_metrics_and_preserves_existing_fields(self):
        A = (0, 1)
        B = (0, 2)
        C = (0, 3)
        D = (0, 4)

        oracle_values = {A: 4.0, B: 3.0, C: 2.0, D: 1.0}
        output = AgentOutput(
            name="integration",
            validated={A: 4.0, B: 3.0, C: 2.0, D: 1.0},
            reported=[A, B, C, D],
            validation_budget=4.0,
        )

        fake_pf = PFResult(
            feasible=True,
            flows={},
            loading={},
            severity=1.0,
            island_penalty=0.0,
            outage=(),
        )

        with patch(
            "poweragentbench.steady_state_agentic.dc_power_flow",
            return_value=fake_pf,
        ):
            result = score_agent(
                original_case=None,
                output=output,
                oracle_values=oracle_values,
                top_m=4,
                danger_threshold=1.0,
            )

        for key in (
            "anytime_risk_auc",
            "anytime_risk_at_25",
            "anytime_risk_at_50",
            "anytime_risk_at_75",
            "anytime_risk_at_100",
        ):
            self.assertIn(key, result)
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

        self.assertAlmostEqual(result["anytime_risk_auc"], 0.625)
        self.assertAlmostEqual(result["anytime_risk_at_100"], 1.0)
        self.assertEqual(result["validated_calls"], 4.0)
        self.assertEqual(result["reported_top20_recall"], 1.0)
        self.assertEqual(result["validated_top20_recall"], 1.0)
        self.assertEqual(result["found_top20_recall"], 1.0)
        self.assertEqual(result["evidence_rate"], 1.0)
        self.assertEqual(result["severity_weighted_false_negative"], 0.0)


class TestToolServerValidationOrder(unittest.TestCase):
    def test_batches_duplicates_and_budget_truncation_preserve_first_completion_order(self):
        A, B, C, D = (0, 1), (0, 2), (0, 3), (0, 4)
        case = type("Case", (), {"name": "stub", "n_bus": 5, "n_line": 4, "n_gen": 1})()

        values = {A: 4.0, B: 3.0, C: 2.0, D: 1.0}
        candidates = [A, B, C, D]

        with patch(
            "poweragentbench.steady_state_agentic.evaluate_contingencies",
            side_effect=lambda _case, selected: {c: values[c] for c in selected},
        ):
            server = SteadyN2ToolServer(case, candidates, validation_budget=3, report_k=3)
            server.execute("validate", {"contingencies": [list(A), list(B)]})
            server.execute("validate", {"contingencies": [list(B), list(C), list(D)]})

        self.assertEqual(list(server.state.validated.keys()), [A, B, C])
        self.assertEqual(server.state.remaining_validations, 0)
        output = AgentOutput(
            name="tool-order",
            validated=server.state.validated,
            reported=[],
            validation_budget=3.0,
        )
        metrics = compute_anytime_risk_metrics(output, values, set(values))
        self.assertAlmostEqual(metrics["anytime_risk_at_100"], 0.90)


if __name__ == "__main__":
    unittest.main()
