#!/usr/bin/env python

from spt3g import maps
import sys
import os

# Test healpix map
# This file was created with this call:
# %> toast_healpix_coadd \
# mapmaker_RISING_SCAN_40-313-4_noiseweighted_map.h5 \
# --nside_submap 8192 \
# --outmap mapmaker_RISING_SCAN_40-313-4_noiseweighted_map_nside8192.fits


def define_splat_projections():
    proj = {}
    proj['spwide'] = {'res': 10*7.27220521664304e-05,
                      'x_len': 2000,
                      'y_len': 2000,
                      'weighted': False,
                      'alpha_center': +0.645772,  # 37
                      'delta_center': -1.179843,  # -67.6
                      'proj': maps.MapProjection.ProjZEA}

    proj['sp0'] = {'res': 7.27220521664304e-05,
                   'x_len': 2400,
                   'y_len': 2400,
                   'weighted': False,
                   'alpha_center': +0.645772,  # 37
                   'delta_center': -0.890118,  # -51
                   'proj': maps.MapProjection.ProjZEA}

    proj['sp1'] = {'res': 7.27220521664304e-05,
                   'x_len': 2400,
                   'y_len': 2400,
                   'weighted': False,
                   'alpha_center': +0.645772,  # 37
                   'delta_center': -1.01229,  # -58
                   'proj': maps.MapProjection.ProjZEA}

    proj['sp2'] = {'res': 7.27220521664304e-05,
                   'x_len': 2400,
                   'y_len': 2400,
                   'weighted': False,
                   'alpha_center': +0.645772,  # 37
                   'delta_center': -1.0821,  # -62
                   'proj': maps.MapProjection.ProjZEA}

    proj['sp3'] = {'res': 7.27220521664304e-05,
                   'x_len': 1920,
                   'y_len': 1920,
                   'weighted': False,
                   'alpha_center': +0.645772,  # 37
                   'delta_center': -1.160644,  # -66.5
                   'proj': maps.MapProjection.ProjZEA}
    return proj


filenames = sys.argv[1:]

projections = define_splat_projections()
proj_name = 'spwide'
proj_name = 'sp1'

print(f"-- Will use projection: {proj_name}")

for filename in filenames:
    print(f"-- Reading into array file: {filename}")
    # We use spt3g, can read gzip files
    hp_array = maps.load_skymap_fits(filename)

    print("-- Transforming Healpix to frame")
    frame3g = maps.healpix_to_flatsky(hp_array['T'], **projections[proj_name])
    print(f"-- New Frame:\n {frame3g}")

    # Write to fits file
    fitsfile = os.path.basename(filename.split('.fits.gz')[0]) + f"_{proj_name}.fits"
    maps.fitsio.save_skymap_fits(fitsfile, frame3g, overwrite=True)
    print(f"-- Wrote FITS:\n {fitsfile}")

print("Done")
