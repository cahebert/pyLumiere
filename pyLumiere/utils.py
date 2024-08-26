import numpy as np
import dust_extinction.parameter_averages

PSEC_TO_CM = 3.085677581e16 * 100

def get_dust_model(dustname, rv):
    """Load dust model from dust_extinction."""
    if dustname == 'O94':
        model = dust_extinction.parameter_averages.O94(Rv=rv)
    elif dustname == 'F19':
        model = dust_extinction.parameter_averages.F19(Rv=rv)
    elif dustname == 'G23':
        model = dust_extinction.parameter_averages.G23(Rv=rv)
    else:
        raise NameError(f"Invalid input: {dustname} not one of O94, F19, G23!")

    # convert to nm from 1/micron
    wlmin = 1e3/model.x_range[1]
    wlmax = 1e3/model.x_range[0]

    return model, wlmin, wlmax
