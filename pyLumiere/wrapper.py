import numpy as np
import galsim
import pystellibs
from . import utils
import astropy.units as u

class Stellib():
    def __init__(self, lib, dustmodel='O94', rv=3.1):
        """"""
        if lib == 'BaSeL':
            self.pystellib = pystellibs.BaSeL()
        elif lib == 'Rauch':
            self.pystellib = pystellibs.Rauch()
        elif lib == 'Kurucz':
            self.pystellib = pystellibs.Kurucz()
        elif lib == 'Tlusty':
            self.pystellib = pystellibs.Tlusty()
        # Elodie library doesn't come with pystellibs distrib?
        # elif lib == 'Elodie':
        #     self.pystellib = pystellibs.Elodie()
        elif lib == 'Munari':
            self.pystellib = pystellibs.Munari()
        elif lib == 'BTSettl':
            self.pystellib = pystellibs.BTSettl(medres=False)
        elif lib == 'Phoenix':
            self.pystellib = pystellibs.Phoenix()
        else:
            raise NameError(f"Invalid input: {lib} not one of BaSeL, Rauch, Kurucz, Tlusty, Munari, BTSettl, or Phoenix!")
        
        self.wl = self.pystellib._wavelength / 10 # convert to nm from Angstrom

        if dustmodel is not None:
            if rv is None:
                raise ValueError('Must provide an Rv value for the dust model!')
            else:
                self.dustmodel = utils.get_dust_model(dustmodel, rv)
        else:
            self.dustmodel = None
        
    def get_intrinsic_sed(self, logte, logg, logl, z):
        """Get 'intrinsic' sed from pystellibs, with units erg/s/A."""
        try:
            return self.pystellib.generate_stellar_spectrum(logte, logg, logl, z)
        except RuntimeError:
            raise ValueError("Input parameters are outside interpolation range!")
    
    def convert_to_observed(self, sed, mu0):
        """Normalize 'intrinsic' sed from erg/s/A to erg/s/A/cm^2. Returns galsim SED object."""
        # need to divide by spherical area given by distance to sun:
        dl = 10**(1 + mu0/5) * utils.PSEC_TO_CM
        sed_obs = sed / (4 * np.pi * dl**2)
    
        sed_table = galsim.LookupTable(self.wl, sed_obs)
        sed_gso = galsim.SED(sed_table, 'nm', 'flambda')
        return sed_gso
    
    def get_dust_extinction(self, av, rv=None, dustmodel=None):
        """Get extinction along stellib wavelengths for given av, rv and dust model.

        If rv, dustmodel parameters are given here they will be used, otherwise the saved 
        dust model (default is O'Donnel 1994 with Rv=3.1, unless changed upon initial
        call to Stellib) will be used."""
        if self.dustmodel is None:
            # no saved model, not enough info provided
            if rv is None or dustmodel is None:
                raise ValueError('Must supply rv AND dustmodel.')
            # no saved model, info provided
            else:
                dust, wlmin, wlmax = utils.get_dust_model(dustmodel, rv)
        else:
            # saved model AND info provided -> use but don't save new info
            if rv is not None and dustmodel is not None:
                dust, wlmin, wlmax = utils.get_dust_model(dustmodel, rv)
                print("FYI: new dustmodel was used but was not saved!")
            # saved model and not enough info provided
            else:
                dust, wlmin, wlmax = self.dustmodel

                # warn that partial info was not used
                if rv is not None or dustmodel is not None:
                    print("Warning: used the saved dustmodel because only one of rv, dustmodel was given!")

        wl_ext = self.wl[np.where((wlmin < self.wl) & (self.wl < wlmax))]
        ext_table = galsim.LookupTable(wl_ext,
                                       dust.extinguish(wl_ext * u.nm, Av=av),
                                       interpolant='linear')
        return galsim.SED(ext_table, wave_type='nm', flux_type='1')

    def get_sed(self, logte, logg, logl, z, mu0, av, rv=None, dustmodel=None):
        """Return extincted observed SED."""
        sed_i = self.get_intrinsic_sed(logte, logg, logl, z)
        sed_o = self.convert_to_observed(sed_i, mu0)

        dust_ext = self.get_dust_extinction(av, rv, dustmodel)

        return sed_o * dust_ext
