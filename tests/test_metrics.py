import unittest
import numpy as np
import pandas as pd
import evaluation.metrics as metrics


class TestDeterministicMetrics(unittest.TestCase):

    def test_absolute_error_for_positive_values(self):
        y_true = pd.array([1, 5, 3])
        y_pred = np.array([0, 2, 10])

        metric = metrics.AbsoluteError()
        AE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_AE = [1, 3, 7]

        self.assertEqual(expected_AE, AE.tolist())

    def test_absolute_error_for_negative_values(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, -2, -10])

        metric = metrics.AbsoluteError()
        AE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_AE = [1, 7, 13]

        self.assertEqual(expected_AE, AE.tolist())

    def test_absolute_percentage_error_for_positive_values(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, 2, 10])

        metric = metrics.AbsolutePercentageError()
        APE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_APE = [100 * 1 / 1, 100 * 3 / 5, 100 * 7 / 3]

        self.assertEqual(expected_APE, APE.tolist())

    def test_absolute_percentage_error_for_negative_values(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, -2, -10])

        metric = metrics.AbsolutePercentageError()
        APE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_APE = [100 * 1 / 1, 100 * 7 / 5, 100 * 13 / 3]

        self.assertEqual(expected_APE, APE.tolist())

    def test_squared_error_for_positive_values(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, 2, 10])

        metric = metrics.RootSquareError()
        SE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_SE = [1 ** 2, 3 ** 2, (-7) ** 2]

        self.assertEqual(expected_SE, SE.tolist())

    def test_squared_error_for_negative_values(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, -2, -10])

        metric = metrics.RootSquareError()
        SE = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_SE = [1 ** 2, 7 ** 2, 13 ** 2]

        self.assertEqual(expected_SE, SE.tolist())

    def test_cumulative_mean(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([0, 2, 10])

        metric = metrics.AbsoluteError()
        AE = metric.calculate_cumulative_error(pd.Series(y_true), pd.Series(y_pred))
        expected_AE = [1 / 1, 4 / 2, 11 / 3]

        self.assertEqual(expected_AE, AE.tolist())

    def test_ramp_score_with_treshold_ignoring_datetimeindex(self):
        y_true = [5, 6.5, 8.5, 7, 6, 5.25]
        y_pred = [5.5, 7, 9, 8, 7, 6.25]
        index = pd.DatetimeIndex([f"2022-04-03 0{i}:00:00" for i in range(6)])

        metric = metrics.RampScore(threshold=2, use_datetime=False)
        RS_dt_ignore = metric.calculate_instant_error(pd.Series(y_true, index=index), pd.Series(y_pred, index=index))
        RS_no_index = metric.calculate_instant_error(pd.Series(y_true), pd.Series(y_pred))
        expected_RS = [0, 0.1, 0.1, 0.1, 0.1, 0.15]
        self.assertEqual(expected_RS, RS_dt_ignore.tolist())
        self.assertEqual(expected_RS, RS_no_index.tolist())

    def test_average_coverage_error_PINC_90_ACE_0(self):
        pi = [0.05, 0.95]
        y_true = np.array([5 for _ in range(10)])
        y_lower = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 7])
        y_upper = np.array([6, 6, 6, 7, 7, 7, 8, 8, 8, 9])
        bounds = np.vstack((y_lower, y_upper)).T

        metric = metrics.CoverageError(pi[0], pi[1])
        ACE = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(bounds, columns=pi))
        expected_ACE = 0
        self.assertEqual(expected_ACE, ACE)

    def test_average_coverage_error_PINC_90_ACE_0_quantile_crossing(self):
        pi = [0.05, 0.95]
        y_true = np.array([5 for _ in range(10)])
        y_lower = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 7])
        y_upper = np.array([6, 6, 6, 7, 7, 7, 8, 8, 8, 4])
        bounds = np.vstack((y_lower, y_upper)).T

        metric = metrics.CoverageError(pi[0], pi[1])
        ACE = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(bounds, columns=pi))
        expected_ACE = 0
        self.assertEqual(expected_ACE, ACE)

    def test_average_winkler_score_PINC_90(self):
        pi = [0.05, 0.95]
        pinc = round(max(pi) - min(pi), 2)
        y_true = np.array([5 for _ in range(10)])
        y_lower = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 7])
        y_upper = np.array([6, 6, 6, 7, 7, 7, 8, 8, 8, 9])
        bounds = np.vstack((y_lower, y_upper)).T

        metric = metrics.WinklerScore(pi[0], pi[1])
        WKS = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(bounds, columns=pi))
        expected_WKS = (9 * 5 + 2 + 2 * 2 / (1 - pinc)) / len(y_true)
        self.assertEqual(expected_WKS, WKS)

    def test_average_winkler_score_PINC_90_quantile_crossing(self):
        pi = [0.05, 0.95]
        pinc = round(max(pi) - min(pi), 2)
        y_true = np.array([5 for _ in range(10)])
        y_lower = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 7])
        y_upper = np.array([6, 6, 6, 7, 7, 7, 8, 8, 8, 4])
        bounds = np.vstack((y_lower, y_upper)).T

        metric = metrics.WinklerScore(pi[0], pi[1])
        WKS = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(bounds, columns=pi))
        expected_WKS = (9 * 5 + (-3) + 2 * 2 / (1 - pinc) + 2 * 1 / (1 - pinc)) / len(y_true)
        self.assertEqual(expected_WKS, WKS)

    def test_pinball_loss_average_five_percent_quantile(self):
        quantiles = [0.05]
        y_true = np.array([50, 50, 50])
        y_pred = np.array([40, 50, 60])

        metric = metrics.PinballLoss(quantiles)
        PBL = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred, columns=quantiles))
        expected_PBL = [(quantiles[0]) * (50 - 40), (quantiles[0]) * (50 - 50), (quantiles[0] - 1) * (50 - 60)]
        self.assertEqual(np.mean(expected_PBL), PBL.mean())

    def test_pinball_loss_average_two_quantiles(self):
        quantiles = [0.05, 0.1]
        y_true = np.array([50, 50, 50])
        y_pred = np.array([[40, 40, 40], [60, 60, 60]])

        metric = metrics.PinballLoss(quantiles)
        PBL = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_PBL = [(0.05 * (50 - 40) + (0.1 - 1) * (50 - 60)) / 2, (0.05 * (50 - 40) + (0.1 - 1) * (50 - 60)) / 2,
                        (0.05 * (50 - 40) + (0.1 - 1) * (50 - 60)) / 2]
        self.assertEqual(expected_PBL, PBL.tolist())

    def test_average_pinball_loss_average(self):
        quantiles = [round(q / 100, 2) for q in range(1, 100)]
        t_steps = 3
        y_true = np.array([50 for _ in range(t_steps)])
        y_pred = np.zeros((t_steps, len(quantiles)))

        count = 1
        for q, quantile in enumerate(quantiles):
            y_pred[:, q] = [count for _ in range(t_steps)]
            count += 1

        metric = metrics.PinballLoss(quantiles)
        PBL = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred, columns=quantiles))
        expected_PBL = 0
        for t in range(t_steps):
            for q, quantile in enumerate(quantiles):
                diff = y_true[t] - y_pred[t, q]
                if diff >= 0:
                    expected_PBL += quantile * diff
                else:
                    expected_PBL += (quantile - 1) * diff
        self.assertEqual(expected_PBL / (len(quantiles) * t_steps), PBL)

    def test_crps_three_quantiles(self):
        quantiles = [0.05, 0.5, 0.1]
        y_true = np.array([50, 50, 50])
        y_pred = np.array([[41, 42, 43], [51, 52, 53], [61, 62, 63]])

        metric = metrics.ContinuousRankedProbabilityScore()
        CRPS = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_CRPS = [(9 + 1 + 11) / 3 - (10 + 10 + 10 + 10) / (3*3),
                         (8 + 2 + 12) / 3 - (10 + 10 + 10 + 10) / (3*3),
                         (7 + 3 + 13) / 3 - (10 + 10 + 10 + 10) / (3*3)]
        self.assertEqual(expected_CRPS, CRPS.tolist())

    def test_quantile_crossing_three_quantiles_crossing(self):
        quantiles = [0.05, 0.1, 0.15]
        y_true = np.array([50, 50, 50])
        y_pred = np.array([[40, 40, 40], [20, 60, 30], [30, 70, 20]])

        metric = metrics.QuantileCrossing(quantiles)
        QC = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_QC = [2 / 3, 0, 3 / 3]
        self.assertEqual(expected_QC, QC.tolist())

    def test_boundary_crossing_left_limit_one_crossing(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([[-1, 0.5, 10], [0.5, 0.5, 0.5]])

        metric = metrics.BoundaryCrossing(0, None)
        BC = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred))
        expected_BC = [1 / 3, 0 / 3]
        self.assertEqual(expected_BC, BC.tolist())

    def test_boundary_crossing_right_limit_one_crossing(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([[-1, 0.5, 10], [0.5, 0.5, 0.5]])

        metric = metrics.BoundaryCrossing(None, 1)
        BC = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred))
        expected_BC = [1 / 3, 0 / 3]
        self.assertEqual(expected_BC, BC.tolist())

    def test_boundary_crossing_both_limit_two_crossings(self):
        y_true = np.array([1, 5, 3])
        y_pred = np.array([[-1, 0.5, 10], [0.5, 0.5, 0.5]])

        metric = metrics.BoundaryCrossing(0, 1)
        BC = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred))
        expected_BC = [2 / 3, 0 / 3]
        self.assertEqual(expected_BC, BC.tolist())

    def test_quantile_crossing_matrix_three_quantiles_crossing(self):
        quantiles = [0.05, 0.1, 0.15]
        y_true = np.array([50, 50, 50])
        y_pred = np.array([[40, 40, 40], [20, 60, 30], [30, 70, 20]])

        metric = metrics.QuantileCrossingMatrix(quantiles)
        QC = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_QC = [4 / 6, 3 / 6, 3 / 6]
        self.assertEqual(expected_QC, QC.tolist())

    def test_interval_width_with_one_quantile_crossing(self):
        lower_bound, upper_bound = 0.05, 0.95
        y_true = np.array([50, 50])
        y_pred = np.array([[40, 30], [20, 50]])

        metric = metrics.IntervalWidth(lower_bound, upper_bound)
        IW = metric.calculate_instant_error(pd.Series(y_true),
                                            pd.DataFrame(y_pred.T, columns=[lower_bound, upper_bound]))
        expected_IW = [20, 20]
        self.assertEqual(expected_IW, IW.tolist())

    def test_calibration_error_with_one_quantile_crossing(self):
        quantiles = [0.05, 0.95]
        y_true = np.array([50, 50])
        y_pred = np.array([[40, 30], [20, 50]])

        metric = metrics.CalibrationError(quantiles)
        CE = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        mean_CE = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_CE = [0 / 2, 1 / 2]
        expected_mean_CE = (abs(0 / 2 - 0.05) + abs(1 / 2 - 0.95)) / 2
        self.assertEqual(expected_CE, CE.tolist())
        self.assertEqual(expected_mean_CE, mean_CE.tolist())

    def test_skewness_with_bowley_variant(self):
        quantiles = [0.25, 0.5, 0.75]
        y_true = np.array([50, 50])
        y_pred = np.array([[20, 30], [35, 47], [40, 50]])

        metric = metrics.Skewness()
        SN = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        mean_SN = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_SN = [(40 + 20 - 2 * 35) / (40 - 20), (50 + 30 - 2 * 47) / (50 - 30)]
        expected_mean_SN = np.mean(expected_SN)
        self.assertEqual(expected_SN, SN.tolist())
        self.assertEqual(expected_mean_SN, mean_SN.tolist())

    def test_kurtosis_with_default_parameters(self):
        quantiles = [0.01, 0.25, 0.75, 0.99]
        y_true = np.array([50, 50])
        y_pred = np.array([[20, 30], [30, 40], [40, 50], [50, 60]])

        metric = metrics.Kurtosis()
        SN = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        mean_SN = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_SN = [(50 - 20) / (40 - 30), (60 - 30) / (50 - 40)]
        expected_mean_SN = np.mean(expected_SN)
        self.assertEqual(expected_SN, SN.tolist())
        self.assertEqual(expected_mean_SN, mean_SN.tolist())

    def test_quartile_dispersion_with_default_parameters(self):
        quantiles = [0.25, 0.75]
        y_true = np.array([50, 50])
        y_pred = np.array([[20, 30], [50, 60]])

        metric = metrics.QuartileDispersion()
        SN = metric.calculate_instant_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        mean_SN = metric.calculate_mean_error(pd.Series(y_true), pd.DataFrame(y_pred.T, columns=quantiles))
        expected_SN = [(50 - 20) / (50 + 20), (60 - 30) / (60 + 30)]
        expected_mean_SN = np.mean(expected_SN)
        self.assertEqual(expected_SN, SN.tolist())
        self.assertEqual(expected_mean_SN, mean_SN.tolist())

if __name__ == '__main__':
    unittest.main()


