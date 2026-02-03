# ----------------------------------------------------------------------------
# 2D IMAGE SUPER-RESOLUTION CONFIGS (High Frequency)
# Used by: main.py
# ----------------------------------------------------------------------------
BEST_CONFIGS = {
    'siren':  {'first_omega_0': 80.0, 'hidden_omega_0': 30.0, 'hidden_layers': 4},
    'mfn':    {'input_scale': 256.0, 'alpha': 6.0, 'beta': 1.0, 'hidden_layers': 4},
    'fourier':{'gamma': 256, 'hidden_layers': 4},
    'gauss':  {'input_scale': 24.0, 'sigma': 1.0, 'hidden_layers': 4},
    'wire':   {'first_omega_0': 10.0, 'hidden_omega_0': 10.0, 'sigma0': 2.5, 'hidden_layers': 4},
    'finer':  {'frequency_bands': 6, 'hidden_layers': 4},
    'incode': {'scale': 256, 'hidden_layers': 4},
    'fr':     {'F': 64,'hidden_layers': 4},
}

# ----------------------------------------------------------------------------
# GRID SEARCH SPACES (2D Super-Resolution)
# Used by: gridsearch.py
# Each key maps to a list of values to try for that hyperparameter.
# Fixed params (like hidden_layers) can also be searched over.
# ----------------------------------------------------------------------------
GRID_SEARCH_SPACES = {
    'siren': {
        'first_omega_0': [30.0, 60.0, 80.0, 120.0],
        'hidden_omega_0': [15.0, 30.0, 60.0],
        'hidden_layers': [3, 4, 5],
    },
    'mfn': {
        'input_scale': [64.0, 128.0, 256.0],
        'alpha': [4.0, 6.0, 8.0],
        'beta': [0.5, 1.0, 2.0],
        'hidden_layers': [3, 4, 5],
    },
    'fourier': {
        'gamma': [64, 128, 256, 512],
        'hidden_layers': [3, 4, 5],
    },
    'gauss': {
        'input_scale': [12.0, 24.0, 48.0],
        'sigma': [0.5, 1.0, 2.0],
        'hidden_layers': [3, 4, 5],
    },
    'wire': {
        'first_omega_0': [5.0, 10.0, 20.0],
        'hidden_omega_0': [5.0, 10.0, 20.0],
        'sigma0': [1.0, 2.5, 5.0],
        'hidden_layers': [3, 4, 5],
    },
    'finer': {
        'frequency_bands': [4, 6, 8, 12],
        'hidden_layers': [3, 4, 5],
    },
    'incode': {
        'scale': [64, 128, 256],
        'hidden_layers': [3, 4, 5],
    },
    'fr': {
        'F': [32, 64, 128],
        'hidden_layers': [3, 4, 5],
    },
}

# ----------------------------------------------------------------------------
# 3D OCCUPANCY CONFIGS (Low Frequency / Smoother)
# Used by: train_3d.py
# ----------------------------------------------------------------------------
BEST_CONFIGS_3D = {
    'siren':  {'first_omega_0': 30.0, 'hidden_omega_0': 30.0, 'hidden_layers': 4},
    'mfn':    {'input_scale': 10.0,  'alpha': 6.0, 'beta': 1.0, 'hidden_layers': 4},
    'fourier': {'gamma': 150.0, 'hidden_layers': 4, 'hidden_features': 512},
    'gauss':  {'input_scale': 20.0, 'sigma': 1.0, 'hidden_layers': 4},
    'wire':   {'first_omega_0': 40.0, 'hidden_omega_0': 20.0, 'sigma0': 3.0, 'hidden_layers': 4},
    'finer':  {'frequency_bands': 16, 'hidden_layers': 4},
    'incode': {'scale': 50.0,'hidden_layers': 4},
    'fr':     {'F': 64, 'hidden_layers': 4},
}
