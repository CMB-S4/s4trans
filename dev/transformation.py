#!/usr/bin/env python

import math


def fac_temp2psflux(nughz, fwhm_arcmin):
    sig_smooth = fwhm_arcmin/math.sqrt(8.*math.log(2.))/(60.*180.)*math.pi
    area = 2.*math.pi*sig_smooth**2
    result = dcmbrdt(nughz*1.e9)/1.e-26*area
    return result


def dbdt(t, f):
    """
    t in K,
    f in Hz returns SI units
    """
    h = 6.6260755e-34
    c = 2.99792458e+8
    k = 1.380658e-23
    x = h*f/(k*t)
    return (2.0*h*f**3/c**2) / (math.exp(x)-1.0)**2 * (x/t) * math.exp(x)


def dcmbrdt(f):
    return dbdt(2.726, f)


def mJy2K_CMB(flux_mJy, freq=150, fwhm=1.2):
    return flux_mJy*1e-3/fac_temp2psflux(freq, fwhm)


# To get the peak temperature fluctuation of a 10 mJy source at 150 GHz in uK
x = 10e-3/fac_temp2psflux(150., 1.2)*1.e6
print(f"{x}[uK]")
print(f"{x/1e6}[K]")
print(f"{mJy2K_CMB(10)}[K]")
