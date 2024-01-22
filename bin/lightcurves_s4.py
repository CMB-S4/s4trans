#!/usr/bin/env python

import argparse as ap

all_bands = ["90GHz", "150GHz", "220GHz"]

P = ap.ArgumentParser(
    description="Fit transient profiles to outlier map pixels for transient detection"
)
P.add_argument("input", nargs="+", help="Input map(s) to include in the fit")
P.add_argument("-o", "--output", help="Output filename")
P.add_argument(
    "--thumb-bands",
    nargs="+",
    default=all_bands,
    choices=all_bands,
    help="Frequency bands to be included in thumbnails",
)
P.add_argument(
    "--fit-bands",
    nargs="+",
    default=all_bands,
    choices=all_bands,
    help="Frequency bands to be included in the fit",
)
P.add_argument(
    "--obs-min",
    default=None,
    type=int,
    help="Minimum observation ID to include in outlier selection",
)
P.add_argument(
    "--obs-max",
    default=None,
    type=int,
    help="Maximum observation ID to include in outlier selection",
)
P.add_argument(
    "-s",
    "--sn-threshold",
    default=2,
    type=float,
    help="Threshold for outlier pixels, in units of map variance",
)
P.add_argument(
    "--no-noise-cut",
    dest="noise_cut",
    action="store_false",
    help="Do not cut maps above some nominal noise chi-square threshold",
)
P.add_argument(
    "-w",
    "--thumb-width",
    default=41,
    type=int,
    help="Width of output thumbnail maps, in pixels",
)
P.add_argument(
    "-a",
    "--algorithm",
    default="simple",
    choices=["simple", "gradient"],
    help="Fitting algorithm to use. The 'simple' algorithm is "
    "Nelder-Mead Simplex2, and 'gradient' is BFGS2.",
)
P.add_argument(
    "-t",
    "--ts-min",
    default=36,
    type=float,
    help="Events with fit test statistic above this threshold will include "
    "thumbnail maps frames in the output file",
)
P.add_argument(
    "-f",
    "--field",
    type=str,
    default=None,
    help="Field name for automatically determining point source file to use.",
)
P.add_argument(
    "--point-source-file",
    type=str,
    default=None,
    help="Pixels within point-source-thresh arcmin of sources in this file "
    "will be excluded from the search.  Ignored if --field is set.",
)
P.add_argument(
    "--point-source-thresh",
    default=5.0,
    type=float,
    help="Distance threshold (in arcmin) to use for point source removal",
)
P.add_argument(
    "--revisit",
    action="store_true",
    help="After the initial analysis, revisit a neighborhood around each "
    "pixel with TS > min TS.",
)
P.add_argument(
    "--revisit-radius",
    default=4.0,
    type=float,
    help="Radius to use for revisiting pixel neighbordhoods.",
)
P.add_argument(
    "--filtered-coadd",
    type=str,
    nargs="+",
    default=[],
    help="Filtered coadd(s) to extract baseline fluxes for each event",
)
P.add_argument(
    "--add-ts-maps",
    action="store_true",
    help="Calculate TS Maps for bright events",
)

P.add_argument("-v", "--verbose", action="store_true", help="Print results")
P.add_argument("-d", "--debug", action="store_true", help="Print debugging messages")

args = P.parse_args()

from spt3g import core, maps, transients

args.revisit_radius *= core.G3Units.arcmin

if args.verbose:
    core.set_log_level(core.G3LogLevel.LOG_INFO, "LightCurve")
elif args.debug:
    args.verbose = True
    core.set_log_level(core.G3LogLevel.LOG_DEBUG, "LightCurve")

# run through input files once to extract outlier pixels
core.log_notice("Extracting outlier pixels...", unit="LightCurve")
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.input)
if args.noise_cut:
    pipe.Add(transients.filter_tools.ApplyMapNoiseCut)
# outliers are pixels that are above this SN threshold in any of the input maps,
# simultaneously in all bands to be included in the fit
pixel_extractor = transients.ExtractMapOutliers(
    map_id=args.fit_bands,
    snthresh=args.sn_threshold,
    sign="same",
    combine_method="any_simultaneous",
    obs_min=args.obs_min,
    obs_max=args.obs_max,
)
pipe.Add(pixel_extractor)

pipe.Run(profile=True)

fit_pixels = pixel_extractor.outliers
if len(fit_pixels) and (args.field is not None or args.point_source_file is not None):
    fit_pixels = transients.lightcurve_tools.remove_pixels_near_sources(
        fit_pixels,
        pixel_extractor.stub,
        field=args.field,
        point_source_file=args.point_source_file,
        point_source_thresh=args.point_source_thresh * core.G3Units.arcmin,
    )
core.log_notice(
    "Found {} outliers in {} maps".format(len(fit_pixels), len(args.input)),
    unit="LightCurve",
)


def extract_baseline_flux(fit_pixels):
    core.log_notice("Extracting baseline fluxes...", unit="LightCurve")
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.filtered_coadd)

    flux_extractor = transients.ExtractFluxes(
        pixels=fit_pixels,
        map_ids=args.fit_bands,
    )
    pipe.Add(flux_extractor)
    pipe.Run(profile=True)
    return flux_extractor.flux


baseline_flux = None
if args.filtered_coadd:
    baseline_flux = extract_baseline_flux(fit_pixels)


def extract_lightcurve_frames(fit_pixels, baseline_flux, noise_cut=True):
    if not len(fit_pixels):
        return {}
    core.log_notice("Extracting light curves...", unit="LightCurve")
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.input)
    if noise_cut:
        pipe.Add(transients.filter_tools.ApplyMapNoiseCut)

    lc_extractor = transients.ExtractLightCurves(
        pixels=fit_pixels,
        map_ids=args.fit_bands,
        baseline_flux=baseline_flux,
        fitter_kwargs={"simple": (args.algorithm == "simple")},
    )
    pipe.Add(lc_extractor)
    pipe.Run(profile=True)
    return lc_extractor.lc_frames


lc_frames = extract_lightcurve_frames(fit_pixels, baseline_flux, noise_cut=args.noise_cut)


# select pixels with test statistics above threshold
thumb_pixels = list(lc_frames)
if args.ts_min is not None:
    thumb_pixels = []
    for pix, lcf in lc_frames.items():
        if lcf["TestStatistic"] > args.ts_min:
            thumb_pixels.append(pix)

    n_thumb = len(thumb_pixels)

    core.log_notice(
        "Found {} pixels with TS > {}".format(n_thumb, args.ts_min),
        unit="LightCurve",
    )

    if args.revisit and len(thumb_pixels):
        import numpy as np

        revisit_pixels = np.array([], dtype=int)
        for pix in thumb_pixels:
            ra, dec = pixel_extractor.stub.pixel_to_angle(pix)
            nearby = pixel_extractor.stub.query_disc(ra, dec, args.revisit_radius)
            revisit_pixels = np.concatenate((revisit_pixels, nearby))

        revisit_pixels = np.setdiff1d(revisit_pixels, fit_pixels)
        if args.field is not None or args.point_source_file is not None:
            revisit_pixels = transients.lightcurve_tools.remove_pixels_near_sources(
                revisit_pixels,
                pixel_extractor.stub,
                field=args.field,
                point_source_file=args.point_source_file,
                point_source_thresh=args.point_source_thresh * core.G3Units.arcmin,
            )

        core.log_notice(
            "Revisiting {} pixels near events with TS > {}".format(
                len(revisit_pixels), args.ts_min
            ),
            unit="LightCurve",
        )

        if len(revisit_pixels):

            revisit_baseline_flux = None
            if args.filtered_coadd:
                revisit_baseline_flux = extract_baseline_flux(revisit_pixels)

            revisit_frames = extract_lightcurve_frames(
                revisit_pixels,
                revisit_baseline_flux,
                noise_cut=args.noise_cut,
            )
            for pix, lcf in revisit_frames.items():
                if lcf["TestStatistic"] > args.ts_min:
                    thumb_pixels.append(pix)
            lc_frames.update(revisit_frames)

        core.log_notice(
            "Found {} additional pixels with TS > {}".format(
                len(thumb_pixels) - n_thumb, args.ts_min
            ),
            unit="LightCurve",
        )

thumb_frames = {}
if len(thumb_pixels):
    # run through input files a third time to extract thumbnails for interesting pixels
    core.log_notice("Extracting thumbnails...", unit="LightCurve")
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=args.input)
    if args.noise_cut:
        pipe.Add(transients.filter_tools.ApplyMapNoiseCut)

    thumb_extractor = transients.ExtractThumbnailMaps(
        pixels=thumb_pixels,
        map_ids=args.thumb_bands,
        thumb_width=args.thumb_width,
    )
    pipe.Add(thumb_extractor)

    pipe.Run(profile=True)

    thumb_frames = thumb_extractor.thumbnails


# collate and write the results to output frames.
core.log_notice("Storing results...", unit="LightCurve")
pipe = core.G3Pipeline()

pipe.Add(
    transients.CollateLightCurves,
    stub=pixel_extractor.stub,
    lc_frames=lc_frames,
    thumb_frames=thumb_frames,
)

if args.add_ts_maps:
    pipe.Add(
        transients.AddTSMaps,
        ts_min=45,
        window=(11, 11),
        fitter_kwargs={"simple": (args.algorithm == "simple")},
    )
    pipe.Add(transients.FindTSCentroid)

if args.debug:
    pipe.Add(core.Dump)

pipe.Add(core.G3Writer, filename=args.output)
pipe.Run(profile=True)
