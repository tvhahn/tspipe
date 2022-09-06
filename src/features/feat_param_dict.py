from itertools import product

dummy_features = {
    "length": None,
    "kurtosis": None,
}

comprehensive_features = {}

comprehensive_features.update({
    # add these simple statistical features
    "mean_abs_change": None,
    "abs_energy": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    "median": None,
    "mean": None,
    "standard_deviation": None,
    "variation_coefficient": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "root_mean_square": None,
    "absolute_sum_of_changes": None,
    "longest_strike_below_mean": None,
    "longest_strike_above_mean": None,
    "count_above_mean": None,
    "count_below_mean": None,
    
    # these are the remaining "comprehensive" features as originally defined
    # by TSFRESH
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    "number_cwt_peaks": [{"n": n} for n in [1, 5]],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
    "index_mass_quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
    "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
                            width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(10 + 1) for k in [10]],
    "change_quantiles": [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
                            for ql in [0., .2, .4, .6, .8] for qh in [.2, .4, .6, .8, 1.]
                            for b in [False, True] for f in ["mean", "var"] if ql < qh],
    "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                        product(["real", "imag", "abs", "angle"], range(100))],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "value_count": [{"value": value} for value in [0, 1, -1]],
    "range_count": [{"min": -1, "max": 1}, {"min": 1e12, "max": 0}, {"min": 0, "max": 1e12}],
    "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
    "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                        {"attr": "slope"}, {"attr": "stderr"}],
    "agg_linear_trend": [{"attr": attr, "chunk_len": i, "f_agg": f}
                            for attr in ["rvalue", "intercept", "slope", "stderr"]
                            for i in [5, 10, 50]
                            for f in ["max", "min", "mean", "var"]],
    "augmented_dickey_fuller": [{"attr": "teststat"}, {"attr": "pvalue"}, {"attr": "usedlag"}],
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "linear_trend_timewise": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                                {"attr": "slope"}, {"attr": "stderr"}],
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "fourier_entropy":  [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "permutation_entropy":  [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    "query_similarity_count": [{"query": None, "threshold": 0.0}],
    "matrix_profile": [{"threshold": 0.98, "feature": f} for f in ["min", "max", "mean", "median", "25", "75"]]
})

custom_1_features = {}

custom_1_features.update({
    # "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    # "c3": [{"lag": lag} for lag in range(1, 4)],
    # "cid_ce": [{"normalize": True}, {"normalize": False}],
    # "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    # "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    # "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
    # "autocorrelation": [{"lag": lag} for lag in range(10)],
    # "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    # "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    # "number_cwt_peaks": [{"n": n} for n in [1, 5]],
    # "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    # "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
    # "index_mass_quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
    # "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
    #                         width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    # "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(10 + 1) for k in [10]],
    # "change_quantiles": [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
    #                         for ql in [0., .2, .4, .6, .8] for qh in [.2, .4, .6, .8, 1.]
    #                         for b in [False, True] for f in ["mean", "var"] if ql < qh],
    "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                        product(["real", "imag", "abs", "angle"], range(100))],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    # "value_count": [{"value": value} for value in [0, 1, -1]],
    # "range_count": [{"min": -1, "max": 1}, {"min": 1e12, "max": 0}, {"min": 0, "max": 1e12}],
    # "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
    "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                        {"attr": "slope"}, {"attr": "stderr"}],
    "agg_linear_trend": [{"attr": attr, "chunk_len": i, "f_agg": f}
                            for attr in ["rvalue", "intercept", "slope", "stderr"]
                            for i in [5, 10, 50]
                            for f in ["max", "min", "mean", "var"]],
    # "augmented_dickey_fuller": [{"attr": "teststat"}, {"attr": "pvalue"}, {"attr": "usedlag"}],
    # "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "linear_trend_timewise": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                                {"attr": "slope"}, {"attr": "stderr"}],
    # "count_above": [{"t": 0}],
    # "count_below": [{"t": 0}],
    # "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    # "fourier_entropy":  [{"bins": x} for x in [2, 3, 5, 10, 100]],
    # "permutation_entropy":  [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    # "query_similarity_count": [{"query": None, "threshold": 0.0}],
    "matrix_profile": [{"threshold": 0.98, "feature": f} for f in ["min", "max", "mean", "median", "25", "75"]]
})