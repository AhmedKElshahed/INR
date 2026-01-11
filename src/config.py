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
# 3D OCCUPANCY CONFIGS (Low Frequency / Smoother)
# Used by: train_3d.py
# ----------------------------------------------------------------------------
BEST_CONFIGS_3D = {
    'siren':  {'first_omega_0': 50.0, 'hidden_omega_0': 30.0, 'hidden_layers': 4},
    'mfn':    {'input_scale': 10.0,  'alpha': 6.0, 'beta': 1.0, 'hidden_layers': 4},
    'fourier':{'gamma': 50.0,'hidden_layers': 4},
    'gauss':  {'input_scale': 10.0, 'sigma': 1.0, 'hidden_layers': 4},
    'wire':   {'first_omega_0': 40.0, 'hidden_omega_0': 20.0, 'sigma0': 3.0, 'hidden_layers': 4},
    'finer':  {'frequency_bands': 16, 'hidden_layers': 4},
    'incode': {'scale': 25.0,'hidden_layers': 4},
    'fr':     {'F': 64, 'hidden_layers': 4},
}
