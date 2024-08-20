import numpy as np
import pylumiere
import astropy.units as u

def test_init():
    """Test initialization of various pystellibs libraries."""
    pylumiere.Stellib('BaSeL')
    pylumiere.Stellib('Rauch')
    pylumiere.Stellib('Kurucz')
    pylumiere.Stellib('Tlusty')
    # pylumiere.Stellib('Elodie')
    pylumiere.Stellib('Munari')
    pylumiere.Stellib('BTSettl')
    # pylumiere.Stellib('Phoenix')

    with np.testing.assert_raises(NameError):
        pylumiere.Stellib('btsettl')

def test_SEDerr():
    """Test errors raised for out-of-bounds stellar parameters."""
    etoiles = pylumiere.Stellib('BTSettl')

    params = {'logte':2,
              'logg':5.5,
              'logl':3,
              'z':0.2
              }
    
    with np.testing.assert_raises(ValueError):
        etoiles.get_intrinsic_sed(**params)

def test_dusterr():
    """Test errors raised for dust extinction mistakes."""
    pylumiere.utils.get_dust_model('O94', 3.1)
    pylumiere.utils.get_dust_model('F19', 3.1)

    with np.testing.assert_raises(NameError):
        pylumiere.utils.get_dust_model('O19', 3.1)

    etoiles = pylumiere.Stellib('BTSettl')

    np.testing.assert_equal(True, etoiles.dustmodel is not None)
    dust_defaults = etoiles.dustmodel[0]
    Av = 0.8
    ext_defaults = etoiles.get_dust_extinction(Av)
    
    etoiles = pylumiere.Stellib('BTSettl', rv=3.1, dustmodel='O94')
    # test default saved model
    np.testing.assert_equal(dust_defaults.extinguish(x=1.2 / u.um, Av=0.9), 
                            etoiles.dustmodel[0].extinguish(x=1.2 / u.um, Av=0.9))

    # this should not change the saved dust model
    ext_new = etoiles.get_dust_extinction(Av, 3.3, 'O94')
    np.testing.assert_equal(dust_defaults.extinguish(x=1.2 / u.um, Av=0.9), 
                            etoiles.dustmodel[0].extinguish(x=1.2 / u.um, Av=0.9))
    # but it should have output something different for extinction
    np.testing.assert_equal(False, ext_new == ext_defaults)
    
def test_reproducibility():
    """Test whether behavior is consistent with pre-computed example."""
    etoiles = pylumiere.Stellib('BTSettl')

    params = {'logte'    : 3.75,
              'logg'     : 5.5,
              'logl'     : 3,
              'z'        : 0.2,
              'mu0'      : 2,
              'av'       : 0.8,
              'rv'       : 3.1,
              'dustmodel': 'O94'}
    
    result = etoiles.get_sed(**params)
    np.testing.assert_equal(1026.6027837219237, result(1000))

if __name__ == '__main__':
    test_init()
    test_SEDerr()
    test_dusterr()
    test_reproducibility()