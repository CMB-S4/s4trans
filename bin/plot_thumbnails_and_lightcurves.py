#!/usr/bin/env python3
from spt3g import core
from spt3g.sources import catalog_matching
import matplotlib as mpl
from matplotlib import pylab as plt
import numpy as np
import os
import sys
mpl.use('agg')

# Load a g3 file with thumbnails and LightCurve
df = sys.argv[1]
try:
    pmax = float(sys.argv[2])
except Exception:
    pmax = 100

print(df)
# Define the location of the output
saveloc = f"{df}_output/"
if not os.path.exists(saveloc):
    os.makedirs(saveloc)
deg = core.G3Units.deg

obsids = {}
fluxes = {}
fluxerrs = {}
thumbnails = {}
prev_ra = []
prev_dec = []
bright_pixels = []


print(f"Doing datafile 1st time: {df}")
for fr in core.G3File(df):
    if fr.type == core.G3FrameType.LightCurve:
        ra = fr['PixelRa']
        dec = fr['PixelDec']
        dists = []
        if prev_ra:
            _, _, dists = catalog_matching.conical_match(ra,
                                                         dec,
                                                         prev_ra,
                                                         prev_dec,
                                                         thresh=3*core.G3Units.arcmin
                                                         )
        if dists:
            prev_ra.append(ra)
            prev_dec.append(dec)
            continue
        prev_ra.append(ra)
        prev_dec.append(dec)
        bright_pixels.append(fr['PixelIndex'])
        #print(fr)
        # If we want to stop after the 1st one
        # break

print('%i bright pixels found: ' % len(bright_pixels))
print(bright_pixels)

# print("ID RA DEC")
# for k in range(len(bright_pixels)):
#     print(bright_pixels[k], prev_ra[k], prev_dec[k])

tscentroids = {}
print(f"Doing datafile 2nd time: {df}")
for fr in core.G3File(df):
    if fr.type == core.G3FrameType.LightCurve:
        if fr['PixelIndex'] in bright_pixels and fr['PixelIndex'] not in tscentroids:
            tscentroids[fr['PixelIndex']] = (fr['PixelRa'], fr['PixelDec'])

    if 'Wunpol' in fr:
        pixel = fr['PixelIndex']
        if pixel not in bright_pixels:
            continue
        band = fr['Id']
        try:
            f = fr['T'][20, 20]/fr['Wunpol'].TT[20, 20]
            w = fr['Wunpol'].TT[20, 20]**(-0.5)
        except Exception:
            continue
        o = fr['ObservationID']
        if pixel not in obsids:
            obsids[pixel] = {}
            fluxes[pixel] = {}
            fluxerrs[pixel] = {}
            thumbnails[pixel] = {}
        if band not in obsids[pixel]:
            obsids[pixel][band] = []
            fluxes[pixel][band] = []
            fluxerrs[pixel][band] = []
            thumbnails[pixel][band] = []
        if int(o) in obsids[pixel][band]:
            continue
        obsids[pixel][band].append(int(o))
        # Bring the values to mJy
        fluxes[pixel][band].append(1000*f/core.G3Units.mJy)
        fluxerrs[pixel][band].append(1000*w/core.G3Units.mJy)
        thumbnails[pixel][band].append(1000*fr['T']/fr['Wunpol'].TT)


print(f"Number of thumbnails: {len(thumbnails)}")

# Print out the coordinates for the thumbnails
coord_file = saveloc+'coords.dat'
coord_plot = saveloc+'ra_dec_plot.png'
fig = plt.figure(figsize=(8, 8))
with open(coord_file, "w") as f:
    for pixel in thumbnails:
        f.write(f"{pixel} {tscentroids[pixel][0]} {tscentroids[pixel][1]}\n")
        plt.scatter(tscentroids[pixel][0], tscentroids[pixel][1], marker='o', c='k')

plt.xlabel('ra[deg]')
plt.ylabel('dec[deg]')
plt.savefig(coord_plot, dpi=300)
plt.close()
print(f"Wrote thumbnail positions to: {coord_file}")
print(f"Wrote ra/dec plot positions to: {coord_plot}")

print("Done with g3 file -- plotting thumbnails and LightCurves now")
# To plot in arcmin
scale = 0.0041666666666667*60
xticks = [0, 8, 16, 24, 32, 40]
newxticks = [f"{int(x*scale):d}" for x in xticks]

bandfig = {'90GHz': 1, '150GHz': 2, '220GHz': 3}
for pixel in thumbnails:

    # Plot the thumbnails
    for band in thumbnails[pixel]:
        numplots = len(thumbnails[pixel][band])
        plt.figure(figsize=(numplots*3, 3))
        pltnum = 1
        sortedobs = np.argsort(obsids[pixel][band])
        for i in range(numplots):
            plt.subplot(1, numplots, pltnum)
            plt.imshow(thumbnails[pixel][band][sortedobs[i]]/core.G3Units.mJy, cmap='viridis',
                       vmin=-0.9*pmax, vmax=0.9*pmax, origin='lower')
            plt.xticks(xticks, newxticks)
            plt.yticks(xticks, newxticks)
            plt.xlabel('x[arcmin]')
            plt.ylabel('y[arcmin]')
            plt.title(obsids[pixel][band][sortedobs[i]])
            plt.tight_layout()
            pltnum += 1
        plt.suptitle(str(pixel)+' '+band)
        th_name = saveloc+'%s_%s_thumbnails.png' % (pixel, band)
        plt.savefig(th_name)
        print(f"saved: {th_name}")
        plt.close()
    plt.figure(figsize=(10, 5))
    plt.title(pixel)

    # Plot the lightcurves
    for band in obsids[pixel]:

        # Time array in hours
        dt = (np.asarray(obsids[pixel][band])-obsids[pixel][band][0])/3600.
        plt.errorbar(dt,
                     fluxes[pixel][band], fluxerrs[pixel][band], marker='o', ls='', label=band)
    plt.grid()
    plt.axhline(0., c='k', ls='--')
    plt.legend()
    plt.xlabel('Time (hr)')
    plt.ylabel('Recovered Amplitude (mJy)')
    lc_name = saveloc+'%s_lightcurve.png' % (pixel)
    plt.savefig(lc_name)
    print(f"saved: {lc_name}")
    plt.close()
    # break
