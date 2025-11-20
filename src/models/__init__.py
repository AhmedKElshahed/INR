from .siren import Siren
from .mfn import GaborMFN, FourierMFN, GaussianMFN
from .wire import Wire
from .finer import FINER
from .incode import INCode
from .fr import FRPaper

def create_model(model_type, in_features=2, out_features=3, hidden_features=256, hidden_layers=4, **kwargs):
    # Note: We pass in_features and out_features to the constructors now
    models = {
        'siren':   lambda: Siren(in_features=in_features, out_features=out_features, 
                                 hidden_features=hidden_features, hidden_layers=hidden_layers,
                                 first_omega_0=kwargs.get('first_omega_0', 30.0),
                                 hidden_omega_0=kwargs.get('hidden_omega_0', 30.0)),
                                 
        'mfn':     lambda: GaborMFN(in_size=in_features, out_size=out_features, 
                                    hidden_size=hidden_features, n_layers=hidden_layers,
                                    input_scale=kwargs.get('input_scale', 256.0),
                                    alpha=kwargs.get('alpha', 6.0),
                                    beta=kwargs.get('beta', 1.0)),
                                    
        'fourier': lambda: FourierMFN(in_size=in_features, out_size=out_features,
                                      hidden_size=hidden_features, n_layers=hidden_layers,
                                      gamma=kwargs.get('gamma', 256),
                                      sigma=kwargs.get('sigma', 10.0)),
                                      
        'gauss':   lambda: GaussianMFN(in_size=in_features, out_size=out_features,
                                       hidden_size=hidden_features, n_layers=hidden_layers,
                                       input_scale=kwargs.get('input_scale', 256.0),
                                       sigma=kwargs.get('sigma', 1.5)),
                                       
        'wire':    lambda: Wire(in_features=in_features, out_features=out_features,
                                hidden_features=hidden_features, hidden_layers=hidden_layers,
                                first_omega_0=kwargs.get('first_omega_0', 30.0),
                                hidden_omega_0=kwargs.get('hidden_omega_0', 10.0),
                                sigma0=kwargs.get('sigma0', 10.0)),
                                
        'finer':   lambda: FINER(in_features=in_features, out_features=out_features,
                                 hidden_features=hidden_features, hidden_layers=hidden_layers,
                                 frequency_bands=kwargs.get('frequency_bands', 6)),
                                 
        'incode':  lambda: INCode(in_features=in_features, out_features=out_features,
                                  hidden_features=hidden_features, hidden_layers=hidden_layers,
                                  scale=kwargs.get('scale', 256)),
                                  
        'fr':      lambda: FRPaper(in_features=in_features, out_features=out_features,
                                   hidden_features=hidden_features, hidden_layers=hidden_layers,
                                   F=kwargs.get('F', 64), include_dc=True),
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}")
    return models[model_type]()
