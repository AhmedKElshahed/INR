from .siren import Siren
from .mfn import GaborMFN, FourierMFN, GaussianMFN
from .wire import Wire
from .finer import FINER
from .incode import INCode
from .fr import FRPaper

def create_model(model_type, hidden_features=256, hidden_layers=4, **kwargs):
    models = {
        'siren':   lambda: Siren(hidden_features=hidden_features, hidden_layers=hidden_layers,
                                 first_omega_0=kwargs.get('first_omega_0', 30.0),
                                 hidden_omega_0=kwargs.get('hidden_omega_0', 30.0)),
        'mfn':     lambda: GaborMFN(hidden_size=hidden_features, n_layers=hidden_layers,
                                    input_scale=kwargs.get('input_scale', 256.0),
                                    alpha=kwargs.get('alpha', 6.0),
                                    beta=kwargs.get('beta', 1.0)),
        'fourier': lambda: FourierMFN(hidden_size=hidden_features, n_layers=hidden_layers,
                                      gamma=kwargs.get('gamma', 256),
                                      sigma=kwargs.get('sigma', 10.0)),
        'gauss':   lambda: GaussianMFN(hidden_size=hidden_features, n_layers=hidden_layers,
                                       input_scale=kwargs.get('input_scale', 256.0),
                                       sigma=kwargs.get('sigma', 1.5)),
        'wire':    lambda: Wire(hidden_features=hidden_features, hidden_layers=hidden_layers,
                                first_omega_0=kwargs.get('first_omega_0', 30.0),
                                hidden_omega_0=kwargs.get('hidden_omega_0', 10.0),
                                sigma0=kwargs.get('sigma0', 10.0)),
        'finer':   lambda: FINER(hidden_features=hidden_features, hidden_layers=hidden_layers,
                                 frequency_bands=kwargs.get('frequency_bands', 6)),
        'incode':  lambda: INCode(hidden_features=hidden_features, hidden_layers=hidden_layers,
                                  scale=kwargs.get('scale', 256)),
        'fr':      lambda: FRPaper(in_features=2, hidden_features=hidden_features, hidden_layers=hidden_layers,
                              out_features=3, F=kwargs.get('F', 64), include_dc=True),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}")
    return models[model_type]()
