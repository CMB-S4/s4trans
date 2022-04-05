#!/usr/bin/env python

from spt3g import maps

# Test healpix map
# This file was created with this call:
# %> toast_healpix_coadd LAT0_CHLAT_split_schedule_5710/mapmaker_RISING_SCAN_40-313-4_noiseweighted_map.h5 --nside_submap 8192 --outmap mapmaker_RISING_SCAN_40-313-4_noiseweighted_map_nside8192.fits

filename = 'mapmaker_RISING_SCAN_40-313-4_noiseweighted_map_nside8192.fits.gz'
print(f"-- Reading into array file: {filename}")

proj1 = {'res': 7.27220521664304e-05,
         'x_len': 18000,
         'y_len': 25000,
         'weighted': False,
         'alpha_center': +1.003564,  # 57.5
         'delta_center': -0.191986,  # -11
         'proj': maps.MapProjection.ProjZEA}

# We use spt3g, can read gzip files
hp_array = maps.load_skymap_fits(filename)

print("-- Transforming Healpix to frame")
frame3g = maps.healpix_to_flatsky(hp_array['T'], **proj1)
print(f"-- New Frame:\n {frame3g}")

# Write to fits file -- does not work -- KeyError: 'ProjNone'
fitsfile = 'mapmaker_RISING_SCAN_40-313-4_noiseweighted_map_nside8192_nmap.fits'
maps.fitsio.save_skymap_fits(fitsfile, frame3g, overwrite=True)
