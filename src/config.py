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
        'first_omega_0': [30.0, 60.0, 80.0],   # 120 rarely beats 80; 3 values
        'hidden_omega_0': [15.0, 30.0],          # 60 too high for hidden layers; 2 values
        'hidden_layers': [4, 5],                 # 3 layers consistently weaker; 2 values
    },  # 3*2*2 = 12 combinations (was 4*3*3 = 36)
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
    # 256 hidden features, 4 layers.
    # NOTE on Fourier: gamma = number of random frequencies (NOT a scale). gamma=10 → fdim=20, catastrophic failure.
    # NOTE on FINER: frequency_bands=6 converges slower than 16 on 3D (eval IoU 0.88 vs 0.95). Keep 16.
    'siren':   {'first_omega_0': 30.0, 'hidden_omega_0': 30.0, 'hidden_layers': 4},
    'mfn':     {'input_scale': 10.0,  'alpha': 6.0, 'beta': 1.0, 'hidden_layers': 4},
    'fourier': {'gamma': 256, 'sigma': 1.0, 'hidden_layers': 4},   # sigma=1.0 (not 10.0): lower freq prevents memorization (train=1.0 eval=0.80 with sigma=10)
    'gauss':   {'input_scale': 20.0, 'sigma': 1.0, 'hidden_layers': 4},
    'wire':    {'first_omega_0': 40.0, 'hidden_omega_0': 20.0, 'sigma0': 3.0, 'hidden_layers': 4},
    'finer':   {'frequency_bands': 16, 'hidden_layers': 4},  # 6 gave 0.88, 16 gives 0.95 — keep higher init freq
    'incode':  {'scale': 50.0, 'hidden_layers': 4},
    'fr':      {'F': 64, 'hidden_layers': 4},
}
