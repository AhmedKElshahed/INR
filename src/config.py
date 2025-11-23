# ----------------------------------------------------------------------------
# 2D IMAGE SUPER-RESOLUTION CONFIGS (High Frequency)
# Used by: main.py
# ----------------------------------------------------------------------------
BEST_CONFIGS = {
    'siren':  {'first_omega_0': 80.0, 'hidden_omega_0': 30.0, 'hidden_layers': 4},
    'mfn':    {'input_scale': 256.0, 'alpha': 6.0, 'beta': 1.0, 'hidden_layers': 4},
    'fourier':{'gamma': 256, 'hidden_layers': 4},
    'gauss':  {'input_scale': 256.0, 'sigma': 1.5, 'hidden_layers': 4},
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
    'siren':  {
        'first_omega_0': 30.0,   # Lowered from 80 for 3D stability
        'hidden_omega_0': 30.0, 
        'hidden_layers': 4
    },
    'mfn':    {
        'input_scale': 10.0,     # Lowered from 256 to prevent noise
        'alpha': 6.0, 
        'beta': 1.0, 
        'hidden_layers': 4
    },
    'fourier':{
        'gamma': 10.0,           # Matches scale of 3D coordinates [-1, 1]
        'hidden_layers': 4
    },
    'gauss':  {
        'input_scale': 10.0,     # Lowered from 256
        'sigma': 4.0,            # Wider Gaussian for better convergence
        'hidden_layers': 4
    },
    'wire':   {
        'first_omega_0': 10.0,   # Lowered for 3D stability
        'hidden_omega_0': 10.0, 
        'sigma0': 5.0,           # Adjusted for 3D
        'hidden_layers': 4
    },
    'finer':  {
        'frequency_bands': 6, 
        'hidden_layers': 4
    },
    'incode': {
        'scale': 10.0,           # Lowered from 256
        'hidden_layers': 4
    },
    'fr':     {
        'F': 32,                 # Lowered frequency count
        'hidden_layers': 4
    },
}
