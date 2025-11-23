from river.neighbors import LazySearch, SWINN

param_grid = {
        'ExtremelyFastDecisionTreeClassifier': {
        'grace_period': [50, 100, 150, 200, 250],
        'max_depth': [None, 1, 2, 3, 4, 5],
        'min_samples_reevaluate': [15, 20, 25, 30],
        'split_criterion': ['gini', 'info_gain', 'hellinger'],
        'delta': [1e-05, 1e-06, 1e-07, 1e-08],
        'tau': [0.03, 0.04, 0.05, 0.07],
        'leaf_prediction': ['mc', 'nb', 'nba'],
        'nb_threshold': [0, 1, 3, 4],
        'min_branch_fraction': [0.08, 0.01, 0.012],
        'max_share_to_split': [0.93, 0.97, 0.98, 0.99],
        'merit_preprune': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },

    'HoeffdingAdaptiveTreeClassifier': {
        'grace_period': [25, 50, 100, 150, 200],
        'max_depth': [None, 1, 2, 3, 4, 5],
        'split_criterion': ['gini', 'info_gain', 'hellinger'],
        'delta': [1e-05, 1e-06, 1e-07, 1e-08],
        'tau': [0.03, 0.04, 0.05, 0.07],
        'leaf_prediction': ['mc', 'nb', 'nba'],
        'nb_threshold': [0, 1, 3, 4],
        'bootstrap_sampling': [True, False],
        'drift_window_threshold': [200, 250, 300, 350],
        'switch_significance': [0.2, 0.4, 0.5, 0.6],
        'binary_split': [True, False],
        'min_branch_fraction': [0.08, 0.01, 0.012],
        'max_share_to_split': [0.93, 0.97, 0.98, 0.99],
        'stop_mem_management': [True, False],
        'merit_preprune': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },
    'HoeffdingTreeClassifier': {
        'grace_period': [25, 50, 100, 150, 200, 250, 300],
        'max_depth': [None, 1, 2, 3, 4, 5],
        'split_criterion': ['gini', 'info_gain', 'hellinger'],
        'delta': [1e-05, 1e-06, 1e-07, 1e-08, 1e-09],
        'tau': [0.03, 0.04, 0.05, 0.07, 0.09],
        'leaf_prediction': ['mc', 'nb', 'nba'],
        'nb_threshold': [0, 1, 3, 4],
        'binary_split': [True, False],
        'min_branch_fraction': [0.08, 0.01, 0.012, 0.014],
        'max_share_to_split': [0.93, 0.97, 0.98, 0.99, 0.995],
        'stop_mem_management': [True, False],
        'remove_poor_attrs': [True, False],
        'merit_preprune': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },
    'KNNClassifier': {
        'n_neighbors': [3, 5, 10, 15],
        'engine': [LazySearch(), SWINN()],
        'weighted': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },
    'AMFClassifier': {
        'n_estimators': [5, 10, 15, 20, 25],
        'use_aggregation': [True, False],
        'dirichlet': [0.1, 0.3, 0.5, 0.7, 1],
        'split_pure': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },
    'ARFClassifier': {
        'n_models': [5, 10, 15, 20, 25],
        'max_features': ['sqrt', 'log2', None],
        'lambda_value': [4, 5, 6, 7, 8],
        'disable_weighted_vote': [True, False],
        'grace_period': [40, 50, 60, 70, 90],
        'split_criterion': ['gini', 'info_gain', 'hellinger'],
        'delta': [1e-05, 1e-06, 1e-07, 1e-08, 1e-09],
        'tau': [0.03, 0.04, 0.05, 0.07, 0.09],
        'leaf_prediction': ['mc', 'nb', 'nba'],
        'nb_threshold': [0, 1, 3, 4],
        'min_branch_fraction': [0.08, 0.01, 0.012, 0.014],
        'max_share_to_split': [0.93, 0.97, 0.98, 0.99, 0.995],
        'remove_poor_attrs': [True, False],
        'merit_preprune': [True, False],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    },
    'ALMAClassifier': {
        'p': [1, 2, 3, 4, 5],
        'alpha': [0.8, 0.9, 1, 1.1, 1.2],
        'B': [1, 1.1111111111111112, 1.2, 1.3, 1.4],
        'C': [1.1, 1.2, 1.3, 1.4142135623730951, 1.5, 1.6],
        'max_window_size': [60, 120, 150, 210],
        'min_window_size': [15, 30, 60]
    }
}
