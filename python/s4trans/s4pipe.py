from spt3g import core, maps, transients
from s4trans import s4tools
import s4trans
import logging
import types
import time
import math
import os
import sys
from tempfile import mkdtemp
import shutil
import magic
import numpy
import datetime
from spt3g.util.maths import gaussian2d
from astropy.coordinates import SkyCoord
import pandas as pd
import healpy as hp

LOGGER = logging.getLogger(__name__)

# Naming template
INDR_OUTNAME = "{tmpdir}/{fileID}_{proj}.{ext}"
FULL_OUTNAME = "{outdir}/{fileID}_{proj}.{ext}"
FRAC_OUTNAME = "{outdir}/{fileID}.{ext}"
BASE_OUTNAME = "{fileID}"
BASEDIR_OUTNAME = "{outdir}/{fileID}"
FILETYPE_EXT = {'FITS': 'fits', 'G3': 'g3.gz'}


class S4pipe:

    """ A Class to run and manage the Header Service"""

    def __init__(self, **keys):
        self.config = types.SimpleNamespace(**keys)

        # Start logging
        self.setup_logging()

        # Define the projections
        self.proj = define_tiles_projection()
        self.proj_names = self.proj.keys()

        # Init the dbhandle
        self.dbhandle = None

        # Define the self.hp_array dict for later
        self.hp_array = {}
        self.hp_array_wgt = {}

        # Read in catalog
        self.read_source_catalog()
        return

    def read_source_catalog(self):
        """ Read in the csv file with ra,dec,flux"""

        # Now we read the observation sequence
        self.obs_seq = s4tools.load_obs_seq()

        try:
            self.config.source_catalog
        except AttributeError:
            self.logger.warning("--source_catalog not an option")
            return

        if self.config.source_catalog is None:
            self.sources_coords = None
            return

        self.logger.info(f"Reading catalog with sources to insert: {self.config.source_catalog}")
        self.sources_coords = pd.read_csv(self.config.source_catalog, skipinitialspace=True)

        return

    def get_flux_scale(self, filename, obs_key, obs_width, proj_name, scan, nsigma=3):

        filename = os.path.basename(filename)
        simID = os.path.basename(filename).split(".")[0]

        # Get the files from where match the scaling
        files = self.get_files_to_insert(obs_key, obs_width, scan, proj_name)
        x = numpy.linspace(-nsigma*obs_width, nsigma*obs_width, 2*obs_width+1)
        g = s4tools.gaussian(x, obs_width)
        # Add a try and execpt
        if simID in files:
            idx = numpy.where(files == simID)[0][0]
            scale = g[idx]
        else:
            self.logger.info(f"{filename} beyond lightcurve, setting scale=0")
            scale = 0
        self.logger.info(f"Flux scale:{scale} for: {simID},{obs_key},w:{obs_width}")
        return scale

    def get_flux_scales(self, filename, proj_name, scan='RISING', nsigma=3):
        scales = []
        for row in self.sources_coords.itertuples():
            scale = self.get_flux_scale(filename, row.obs_key, row.obs_width, proj_name,
                                        scan=row.scan, nsigma=nsigma)
            scales.append(scale)
        return numpy.asarray(scales)

    def check_input_files(self):
        " Check if the inputs are a list or a file with a list"

        t = magic.Magic(mime=True)
        if self.config.nfiles == 1 and t.from_file(self.config.files[0]) == 'text/plain':
            self.logger.info(f"{self.config.files[0]} is a list of files")
            # Now read them in
            with open(self.config.files[0], 'r') as f:
                lines = f.read().splitlines()
            self.logger.info(f"Read: {len(lines)} input files")
            self.config.files = lines
            self.config.nfiles = len(lines)
        else:
            self.logger.info("Nothing to see here")

    def setup_logging(self):
        """ Simple logger that uses configure_logger() """
        # Create the logger
        s4tools.create_logger(level=self.config.loglevel,
                              log_format=self.config.log_format,
                              log_format_date=self.config.log_format_date)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging Started at level:{self.config.loglevel}")
        self.logger.info(f"Running s4trans version: {s4trans.__version__}")

    def load_healpix_map(self, filename, noweight=False, frame='T'):
        """Load a healpix map as an array"""
        t0 = time.time()
        if filename in self.hp_array:
            self.logger.info(f"Healpix map from: {filename} already stored -- skipping")
        else:
            self.logger.info(f"Reading into array file: {filename}")
            hp_array = maps.load_skymap_fits(filename)
            LOGGER.info(f"Read time: {s4tools.elapsed_time(t0)}")
            self.hp_array[filename] = hp_array[frame]

            # Stop here if no weights is True
            if noweight:
                return

            # Now we read in the weight (incov) file
            vals = filename.split('.fits')
            filename_wgt = f"{vals[0]}_invcov.fits{vals[1]}"
            if os.path.isfile(filename_wgt):
                self.logger.info(f"Reading INCOV into array file: {filename_wgt}")
                incov = hp.read_map(filename_wgt, dtype=numpy.float32, verbose=False)
                # Normalize the weight
                if self.config.normalize_weight:
                    self.logger.info("Normalizing INCOV array")
                    incov = incov/numpy.max(incov)
                self.hp_array_wgt[filename] = numpy.sqrt(incov)
            else:
                raise Exception(f"File {filename_wgt} does not exist")

    def write_fits(self, g3frame, file, proj_name, hdr=None):
        # Get self.outname and self.outname_tmp
        self.set_outname(file, proj_name, filetype='FITS')
        self.logger.info(f"Will create: {self.outname_tmp}")
        t0 = time.time()
        maps.fitsio.save_skymap_fits(self.outname_tmp,
                                     g3frame['T'], W=g3frame['Wunpol'],
                                     compress='GZIP_2', overwrite=True, hdr=hdr)
        # In case we have indirect_write
        self.move_outname()
        self.logger.info(f"FITS file creation time: {s4tools.elapsed_time(t0)}")

    def write_g3(self, g3frame, file, proj_name):
        # Get self.outname and self.outname_tmp
        self.set_outname(file, proj_name, filetype='G3')
        self.logger.info(f"Will create: {self.outname_tmp}")
        t0 = time.time()
        core.G3Writer(filename=self.outname_tmp)(g3frame)
        # In case we have indirect_write
        self.move_outname()
        self.logger.info(f"G3 file creation time: {s4tools.elapsed_time(t0)}")

    def set_outname(self, filename, proj, filetype='FITS'):
        """ Define the name of the projected output files"""
        basename = os.path.basename(filename)
        fileID = basename.split('.')[0]
        ext = FILETYPE_EXT[filetype]
        outdir = self.config.outdir
        self.fileID = fileID
        if self.config.indirect_write:
            tmpdir = self.tmpdir

        kw = locals()
        self.outname = FULL_OUTNAME.format(**kw)
        self.logger.debug(f"Will write: {self.outname}")

        if self.config.indirect_write:
            self.outname_tmp = INDR_OUTNAME.format(**kw)
            self.logger.debug(f"Will use tempfile: {self.outname_tmp}")
        else:
            self.outname_tmp = self.outname

    def set_outname_frac(self, filename, ext='log'):
        """ Define the name of the projected output files"""
        basename = os.path.basename(filename)
        fileID = basename.split('.')[0]
        outdir = self.config.outdir
        self.fileID = fileID

        kw = locals()
        self.outname_frac = FRAC_OUTNAME.format(**kw)
        self.logger.info(f"Will write fractions to: {self.outname_frac}")

    def move_outname(self):
        """Move to original name in case that we use indirect_write"""
        if self.config.indirect_write:
            self.logger.info(f"Moving {self.outname_tmp} --> {self.outname}")
            shutil.move(self.outname_tmp, self.outname)

    def find_onfraction(self):
        """Find the fraction of non zero pixels in projection"""

        t0 = time.time()

        # Check input files vs file list
        self.check_input_files()
        # Make sure that the folder exists: n
        s4tools.create_dir(self.config.outdir)

        nfile = 1
        for file in self.config.files:

            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")
            # Load up the gzip healpix map
            t1 = time.time()
            self.load_healpix_map(file, noweight=True)
            # Get the name of the output file to record the fractions
            self.set_outname_frac(file)
            ofile = open(self.outname_frac, 'w')

            if self.config.proj_name is not None:
                proj_names = self.config.proj_name
            else:
                proj_names = self.proj_names

            k = 1
            for proj_name in proj_names:

                proj = self.proj[proj_name]
                self.logger.info(f"Transforming Healpix to G3 frame for projection: {proj_name}")
                t2 = time.time()
                frame3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
                self.logger.info(f"Done healpix_to_flatsky: {s4tools.elapsed_time(t2)} ")
                on_fraction = frame3g.npix_nonzero/frame3g.size
                sfraction = f"{self.fileID} {proj_name} {on_fraction:.6}"
                self.logger.info(f"FRACTION: {sfraction}")
                ofile.write(sfraction + "\n")
                k += 1

            self.logger.info(f"Done with {file} time: {s4tools.elapsed_time(t1)} ")
            nfile += 1

            # Clean up
            del frame3g
            del self.hp_array[file]

        ofile.close()
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")

    def calculate_onfraction(self, file, proj_name):
        """Compute on fraction for file and proj_name"""
        self.logger.info(f"Transforming Healpix to G3 frame for projection: {proj_name}")
        self.load_healpix_map(file)
        proj = self.proj[proj_name]
        frame3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
        on_fraction = frame3g.npix_nonzero/frame3g.size
        return on_fraction

    def filter_sim_file(self, file, proj_name, filetypes, band='150GHz'):
        """Filter Simulations using transient filtering method"""

        # Get the obs_id for the file and obs_seq
        obs_id = get_obs_id(file, self.obs_seq)

        t0 = time.time()
        proj = self.proj[proj_name]
        self.load_healpix_map(file)
        t1 = time.time()
        self.logger.info(f"Transforming Healpix to G3 frame for projection {proj_name}")
        map3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")
        self.logger.info(f"New Frame:\n {map3g}")

        t1 = time.time()
        map3gTT = maps.healpix_to_flatsky(self.hp_array_wgt[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")
        self.logger.info(f"New Frame:\n {map3gTT}")

        # Insert if catalog is present
        if self.sources_coords is not None:
            scale = self.get_flux_scales(file, proj_name, scan='RISING', nsigma=3)
            flux_scaled = self.sources_coords['FLUX']*scale
            map3g = insert_sources(map3g, self.sources_coords, flux_scaled, norm=False)
            self.logger.info("Done inserting sources")

        # Create frame for output
        g3frame = core.G3Frame(core.G3FrameType.Map)
        g3frame['T'] = map3g
        g3frame['T'].sparse = False

        # Create a weights maps of ones -- fake weights
        if self.config.clone_weight:
            self.logger.info(f"Clonning weights for: {file}")
            weights_clone = map3g.clone()
            idx = numpy.where(weights_clone != 0)
            numpy.asarray(weights_clone)[idx] = 1
            weightmap = maps.G3SkyMapWeights()
            weightmap.TT = weights_clone
        # Now we want to put the data/weights in g3/frame format
        else:
            g3frame = core.G3Frame(core.G3FrameType.Map)
            g3frame['T'] = map3g
            g3frame['T'].sparse = False
            weightmap = maps.G3SkyMapWeights(g3frame['T'], polarized=False)
            weightmap.TT = map3gTT

        # Add to the frame
        g3frame["Wunpol"] = weightmap

        self.logger.info("Loading pipe for filtering")
        pipe = core.G3Pipeline()
        pipe.Add(core.G3InfiniteSource, n=0)
        pipe.Add(maps.InjectMaps, map_id=band, maps_in=[map3g, weightmap])

        pipe.Add(addid_to_frame, obs_id=obs_id)
        pipe.Add(maps.map_modules.MakeMapsUnpolarized)
        pipe.Add(transients.TransientMapFiltering,
                 bands=[band],  # or just band
                 subtract_coadd=False)

        if 'G3' in filetypes:
            # Get the outname
            self.set_outname(file, f"flt_{proj_name}", filetype='G3')
            self.logger.info(f"Preparing to write G3: {self.outname_tmp}")
            pipe.Add(core.G3Writer, filename=self.outname_tmp)

        # Write as FITS file
        if 'FITS' in filetypes:
            # Get the outname
            self.set_outname(file, f"flt_{proj_name}", filetype='FITS')
            self.logger.info(f"Preparing to write FITS: {self.outname_tmp}")
            # We want the unweighted maps
            pipe.Add(maps.RemoveWeights, zero_nans=True)
            pipe.Add(remove_units, units=core.G3Units.mJy)
            hdr = {}
            hdr['OBSID'] = (obs_id, 'ObservationID')
            pipe.Add(maps.fitsio.SaveMapFrame, output_file=self.outname_tmp,
                     compress='GZIP_2', overwrite=True, hdr=hdr)

        self.logger.info("Executing .Run()")
        pipe.Run(profile=True)
        del pipe

        # In case we have indirect_write
        if 'FITS' in filetypes:
            # Get the outname
            self.set_outname(file, f"flt_{proj_name}", filetype='FITS')
            self.move_outname()
            self.logger.info(f"Created file: {self.outname}")

        if 'G3' in filetypes:
            # Get the outname
            self.set_outname(file, f"flt_{proj_name}", filetype='G3')
            self.move_outname()
            self.logger.info(f"Created file: {self.outname}")

        self.logger.info(f"Filtering file {file} done: {s4tools.elapsed_time(t0)} ")

        del map3g
        del map3gTT
        self.clean_up(file)

        return

    def filter_sims(self, filetypes=['FITS', 'G3']):
        """Run function self.filter_sim_file for a set of files and filters"""
        t0 = time.time()
        # Check input files vs file list
        self.check_input_files()

        # Make sure that the folder exists: n
        s4tools.create_dir(self.config.outdir)

        if self.config.proj_name[0] == 'all':
            self.config.proj_name = self.proj_names
            self.logger.info("Will use all proj_names")

        if self.config.indirect_write:
            self.tmpdir = mkdtemp(prefix=self.config.indirect_write_prefix)
            self.logger.info(f"Preparing folder for indirect_write: {self.tmpdir}")
            # Make sure that the folder exists:
            s4tools.create_dir(self.tmpdir)

        nfile = 1
        for file in self.config.files:
            t1 = time.time()
            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")
            for proj_name in self.config.proj_name:
                on_fraction = self.get_db_onfraction(file, proj_name)
                if on_fraction is None:
                    on_fraction = 0
                    self.logger.warning(f"Cannot find fraction for {file} on DB")
                    self.logger.warning(f"Will try to calculate fraction for: {proj_name}")
                    on_fraction = self.calculate_onfraction(file, proj_name)
                    self.logger.info(f"Computed fraction: {on_fraction}")
                if on_fraction > self.config.onfracion_thresh and on_fraction is not None:
                    self.logger.info(f"Fraction of non-zero pixels is ABOVE threshold: {on_fraction}")
                    self.filter_sim_file(file, proj_name, filetypes)
                else:
                    self.logger.info(f"Fraction of non-zero pixels is BELOW threshold: {on_fraction}")
                    self.logger.info(f"Skipping proj:{proj_name} for file: {file}")

            self.logger.info(f"Done with {file} time: {s4tools.elapsed_time(t1)} ")
            nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")
        return

    def project_sims(self, filetypes=['FITS', 'G3']):

        t0 = time.time()
        # Check input files vs file list
        self.check_input_files()

        # Make sure that the folder exists: n
        s4tools.create_dir(self.config.outdir)

        if self.config.proj_name[0] == 'all':
            self.config.proj_name = self.proj_names
            self.logger.info("Will use all proj_names")

        if self.config.indirect_write:
            self.tmpdir = mkdtemp(prefix=self.config.indirect_write_prefix)
            self.logger.info(f"Preparing folder for indirect_write: {self.tmpdir}")
            # Make sure that the folder exists:
            s4tools.create_dir(self.tmpdir)

        nfile = 1
        for file in self.config.files:
            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")

            t1 = time.time()
            for proj_name in self.config.proj_name:
                on_fraction = self.get_db_onfraction(file, proj_name)
                if on_fraction is None:
                    on_fraction = 0
                    self.logger.warning(f"Cannot find fraction for {file} on DB")
                    self.logger.warning(f"Will try to calculate fraction for: {proj_name}")
                    on_fraction = self.calculate_onfraction(file, proj_name)
                    self.logger.info(f"Computed fraction: {on_fraction}")
                # Project if above threshold
                if on_fraction > self.config.onfracion_thresh and on_fraction is not None:
                    self.logger.info(f"Fraction of non-zero pixels is ABOVE threshold: {on_fraction}")
                    self.project_sim_file(file, proj_name, filetypes)
                else:
                    self.logger.info(f"Fraction of non-zero pixels is BELOW threshold: {on_fraction}")
                    self.logger.info(f"Skipping FITS for projection: {proj_name}")

            self.logger.info(f"Done with {file} time: {s4tools.elapsed_time(t1)} ")
            nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")

    def project_sim_file(self, file, proj_name, filetypes):
        """ Project single sim file"""

        # Get the obs_id for the file and obs_seq
        obs_id = get_obs_id(file, self.obs_seq)

        t0 = time.time()
        proj = self.proj[proj_name]
        self.load_healpix_map(file)
        t1 = time.time()
        self.logger.info(f"Transforming Healpix to G3 frame for projection: {proj_name}")
        map3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")
        self.logger.info(f"New Frame:\n {map3g}")

        t1 = time.time()
        map3gTT = maps.healpix_to_flatsky(self.hp_array_wgt[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")
        self.logger.info(f"New Frame:\n {map3gTT}")

        # Insert if catalog is present
        if self.sources_coords is not None:
            scale = self.get_flux_scales(file, proj_name, scan='RISING', nsigma=3)
            flux_scaled = self.sources_coords['FLUX']*scale
            map3g = insert_sources(map3g, self.sources_coords, flux_scaled, norm=False)
            self.logger.info("Done inserting sources")

        # Create frame for output
        g3frame = core.G3Frame(core.G3FrameType.Map)
        g3frame['T'] = map3g
        g3frame['T'].sparse = False

        # Create a weights maps of ones -- fake weights
        if self.config.clone_weight:
            self.logger.info(f"Clonning weights for: {file}")
            weights_clone = map3g.clone()
            idx = numpy.where(weights_clone != 0)
            numpy.asarray(weights_clone)[idx] = 1
            weightmap = maps.G3SkyMapWeights()
            weightmap.TT = weights_clone
        # Now we want to put the data/weights in g3/frame format
        else:
            weightmap = maps.G3SkyMapWeights(g3frame['T'], polarized=False)
            weightmap.TT = map3gTT

        # Add to the frame
        g3frame["Wunpol"] = weightmap

        if 'FITS' in filetypes:
            self.logger.info(f"Preparing to write FITS: {file}")
            hdr = {}
            hdr['OBSID'] = (obs_id, 'ObservationID')
            self.write_fits(g3frame, file, proj_name, hdr)
        if 'G3' in filetypes:
            self.logger.info(f"Preparing to write G3: {file}")
            self.write_g3(g3frame, file, proj_name)
        self.logger.info(f"Projecting of file: {file} done: {s4tools.elapsed_time(t0)} ")

        del map3g
        del map3gTT
        self.clean_up(file)

        return

    def clean_up(self, file):
        del self.hp_array[file]
        del self.hp_array_wgt[file]

    def ingest_onfraction(self):
        """Ingest the on fraction log files into the sqlite3 database"""

        t0 = time.time()
        # Make the connection to the DB
        con = s4tools.connect_db(self.config.dbname, self.config.tablename)

        # Check if we want to replace entry
        replace_insert = self.config.replace_insert

        # Loop over all of the files
        nfile = 1
        for filename in self.config.files:
            t1 = time.time()
            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")
            s4tools.ingest_fraction_file(filename, self.config.tablename,
                                         con=con, replace=replace_insert)
            self.logger.info(f"Done with {filename} time: {s4tools.elapsed_time(t1)} ")
            nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")

    def get_db_onfractions(self, proj_name):

        "Get the on fraction"
        if self.dbhandle is None:
            self.dbhandle = s4tools.connect_db(self.config.dbname)

        # Clean up the name to get the SIMID
        query = f'select SIMID, fraction from {self.config.tablename}'
        query = query + f' where PROJ="{proj_name}"'
        # query = query + f' where SIMID like "%{scan}%" and PROJ="{proj_name}"'
        # query = query + f' and fraction>{self.config.onfracion_thresh}'
        self.logger.info(f"Will run query:\n\t{query}")
        rec = s4tools.query2rec(query, self.dbhandle)
        return rec

    def get_db_onfraction(self, filename, proj_name, verb=True):

        "Get the on fraction"
        if self.dbhandle is None:
            self.dbhandle = s4tools.connect_db(self.config.dbname)

        # Clean up the name to get the SIMID
        simID = os.path.basename(filename).split(".")[0]
        query = f'select fraction from {self.config.tablename} where SIMID="{simID}" and PROJ="{proj_name}" '

        # Get the cursor from the DB handle
        cur = self.dbhandle.cursor()
        # Execute
        cur.execute(query)
        try:
            fraction = cur.fetchone()[0]
            if verb:
                self.logger.info(f"Fraction: {fraction} for {filename}")
        except TypeError:
            fraction = None
            if verb:
                self.logger.warning(f"on-fraction not found for {filename} and {proj_name}")
        cur.close()
        return fraction

    def select_files_threshold(self, obs_key, obs_width, scan, proj_name):
        """
        Get the files above fraction thresold
        """
        # Read the observation sequence only once
        try:
            obs_seq = self.obs_seq
            self.logger.info("Observing sequence already loaded -- skipping")
        except Exception:
            obs_seq = s4tools.load_obs_seq()

        # Select only files above fraction threshold
        try:
            rec = self.rec_onfractions
            self.logger.info(f"On fractions computed for thresh: {self.config.onfracion_thresh} -- skipping")
        except Exception:
            self.rec_onfractions = self.get_db_onfractions(proj_name)
            self.logger.info(f"Computing on fraction for thresh: {self.config.onfracion_thresh}")
            rec = self.rec_onfractions

        # Now we match using the observing sequence order and select
        # based on fraction threshold
        selection = []
        nsequence = len(obs_seq[scan]['filename'])
        for i in range(nsequence):
            file = obs_seq[scan]['filename'][i]
            seqid = obs_seq[scan]['obs_seq'][i]
            simID = os.path.basename(file).split(".")[0]
            # Match by SIMID for fraction
            idx = numpy.where(rec['SIMID'] == simID)[0][0]
            fraction = rec['FRACTION'][idx]
            if fraction > self.config.onfracion_thresh:
                selection.append((simID, seqid, fraction))

        # Recast the selected data as a numpy record array
        data_selected = numpy.rec.array(selection, names=['id', 'obs_seq', 'fraction'])
        return data_selected

    def get_files_to_insert(self, obs_key, obs_width, scan, proj_name):
        """
        Get the files to insert souces that match the fraction threshold
        """

        # Get the filtered universe of files we want to select from
        data_selected = self.select_files_threshold(obs_key, obs_width, scan, proj_name)

        # Get the index for the file with the peak flux
        k = numpy.where(data_selected['obs_seq'] == obs_key)[0][0]

        # Select all files between peak_flux-width and peak_flux+width
        files = data_selected['id'][k-obs_width: k+obs_width+1]
        self.logger.info(f"Found nfiles:{len(files)} for {obs_key},{obs_width},{scan},{proj_name}")
        return files


def define_tiles_projection(x_len=5000, y_len=5000,
                            delta_up=0, delta_low=-50,
                            res=7.27220521664304e-05,  # arcmin?
                            weighted=False):

    d2r = math.pi/180.  # degrees to radians shorthand
    delta_width = round(res*60*y_len, -1)
    LOGGER.info(f"Will define tiles with delta_width: {delta_width} degrees")

    ntiles = 0
    delta_done = False
    proj = {}
    j = 0
    delta_center = delta_up - delta_width/2.0
    while delta_done is False:
        i = 0
        alpha_width = delta_width/math.cos(delta_center*d2r)
        alpha_center = alpha_width/2.0
        while alpha_center < 360:
            proj_name = f"proj_{j:02d}-{i:02d}"
            msg = f"Defining (alpha, delta) center for {proj_name} at {alpha_center:.1f},{delta_center} deg"
            LOGGER.info(msg)
            p = {'res': res,
                 'x_len': x_len,
                 'y_len': y_len,
                 'weighted': weighted,
                 'alpha_center': alpha_center,
                 'delta_center': delta_center*d2r,
                 'proj': maps.MapProjection.ProjZEA}
            proj[proj_name] = p
            alpha_center = i*alpha_width + alpha_width/2.0
            ntiles = ntiles + 1
            i = i + 1

        j = j + 1
        delta_center = delta_up - j*delta_width - delta_width/2.0
        if delta_center <= delta_low:
            delta_done = True
    LOGGER.info(f"Defined {ntiles} tiles")
    # Adding wide test projection
    # delta_center = -30.0020833
    proj_name = 'wide'
    delta_center = -15.0
    alpha_center = 80.8290376865476
    proj[proj_name] = {'res': res,
                       'x_len': 18000,
                       'y_len': 25000,
                       'weighted': weighted,
                       'alpha_center': alpha_center,  # 57.5
                       'delta_center': delta_center*d2r,  # -11
                       'proj': maps.MapProjection.ProjZEA}
    msg = f"Defining (alpha, delta) center for {proj_name} at {alpha_center:.1f},{delta_center} deg"

    # We also a add a small patch for testing
    proj_name = 'small'
    delta_center = -30.0
    alpha_center = 80.8290376865476
    proj[proj_name] = {'res': res,
                       'x_len': 1000,
                       'y_len': 1000,
                       'weighted': weighted,
                       'alpha_center': alpha_center,  # 57.5
                       'delta_center': delta_center*d2r,  # -11
                       'proj': maps.MapProjection.ProjZEA}

    msg = f"Defining (alpha, delta) center for {proj_name} at {alpha_center:.1f},{delta_center} deg"
    LOGGER.info(msg)

    LOGGER.info("Adding defintions for SPLAT regions")

    # Adding definitions for SPLAT regions
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


def define_tiles_projection_old(ntiles=6, x_len=14000, y_len=20000,
                                res=7.27220521664304e-05,
                                weighted=False,
                                delta_center=-0.401834135):  # -23.024 in radians

    LOGGER.info(f"Will define {ntiles} tiles")
    proj = {}
    dx = int(360./ntiles)
    for i in range(ntiles):
        proj_name = f"proj{i+1}"
        LOGGER.info(f"Defining alpha_center for tile {i+1}  at {i*dx} degrees")
        alpha_center = i*dx*math.pi/180.
        p = {'res': res,
             'x_len': x_len,
             'y_len': y_len,
             'weighted': weighted,
             'alpha_center': alpha_center,
             'delta_center': delta_center,
             'proj': maps.MapProjection.ProjZEA}
        proj[proj_name] = p

    return proj


def insert_sources(frame, coords, flux, norm=True, sigma=1.5, nsigma=3):

    """
    Inserts source at (ra,dec) using wcs information from frame
    If norm=True flux is spread over the kernel, otherwise it's the peak value
    at xo,yo. The code relies on the astropy.wcs object contained in the frame
    get the pixel coordinates.

    Arguments:
    ---------
    frame : 3g frame object
        The frame where we want to inject/insert point sourcers into.
    coords: pandas object
        with RA/DEC or XIMAGE/YIMAGE (floats)
    flux: float or list
        The list or float with total flux of the source (units mJy)
    norm: Bool
        Normalize fluxe
    sigma: float
        The size of sigma
    nsigma: float
        The number of sigma to extend for the kernel
    """

    try:
        ra = coords['RA']
        dec = coords['DEC']
        # Get the pixel coordinates
        sky = SkyCoord(ra, dec, unit='deg')
        ximage, yimage = frame.wcs.world_to_pixel(sky)

    except Exception:
        ximage = coords['XIMAGE']
        yimage = coords['YIMAGE']

    if len(ximage) == len(yimage) and len(ximage) == len(flux):
        pass
    else:
        LOGGER.error("Error: xo,yo (ra, dec) and flux must be same length")
        sys.exit()

    # Get the dimensions of the kernel, same for all objects we want to insert
    # Taken from:
    # https://github.com/SouthPoleTelescope/spt3g_software/blob/2a4ab81ba8ef0dd5a939b84fbbce3c76e1c99c35/transients/python/filter_tools.py#L606
    # fwhms = {"90GHz": 1.54 * u, "150GHz": 1.13 * u, "220GHz": 1.0 * u}
    # u = core.G3Units.arcmin
    # fwhm = 1.0*u  # Assuming ~150GHz
    # sigma = fwhms["150GHz"] / 2.35482 / frame3g.res
    # ---------------------------------
    # For now we'll just use sigma=1 and nsigma=3 as defaults in kwargs
    dim = int(numpy.ceil(nsigma * sigma))
    y, x = numpy.indices((2 * dim + 1, 2 * dim + 1))
    nobjects = len(ximage)
    LOGGER.info(f"Will try to insert {nobjects} object(s) in 3G frame")
    for k in range(nobjects):

        # Need to re-cast astropy/numpy objects as scalars -- and round
        xo = round(float(ximage[k]))
        yo = round(float(yimage[k]))

        # Transform the flux from mJy to K_CMB
        flux_cmb = s4tools.mJy2K_CMB(flux[k], freq=150, fwhm=1.0)

        # Generate the kernel with fluxes
        kernel = gaussian2d(flux_cmb, dim, dim, sigma, norm=norm)(y, x)

        # Ignore when flux == 0
        if flux_cmb == 0:
            LOGGER.info(f"Skipping {dim,dim} kernel at {xo,yo} with flux {flux_cmb}")
            continue

        # Now inject the kernel at xo,yo
        LOGGER.info(f"Inserting {dim,dim} kernel at {xo,yo} with flux {flux_cmb}")
        orig_data = numpy.asarray(frame)[yo-dim:yo+dim+1, xo-dim:xo+dim+1]
        numpy.asarray(frame)[yo-dim:yo+dim+1, xo-dim:xo+dim+1] = orig_data + kernel

    return frame


def addid_to_frame(fr, obs_id):
    """ Add obs_id to a frame"""
    fr['ObservationID'] = obs_id


def get_obs_id(file, obs_seq):
    """ Common method to get obs_id based on the name of the file"""
    date0 = datetime.datetime(2023, 1, 1, 0, 0).timestamp()
    f = os.path.basename(file)
    # Check if RISING, SETTING or POLE
    if f.find('RISING') > 0:
        scan = 'RISING'
    elif f.find('SETTING') > 0:
        scan = 'SETTING'
    elif f.find('POLE') > 0:
        scan = 'POLE'
    else:
        LOGGER.error(f"Cannot find RISING/SETTING/POLE from file: {f}")
        sys.exit(1)
    idx = numpy.where(obs_seq[scan]['filename'] == f)[0][0]
    if scan == 'RISING':
        dtime = (2*idx + 1)*60
    if scan == 'SETTING':
        dtime = (2*idx + 2)*60
    if scan == 'POLE':
        dtime = (2*idx - 1)*60

    obs_id = int(date0 + dtime)
    LOGGER.info(f"Will add obs_id: {obs_id} to: {file}")
    LOGGER.info(f"Date: {datetime.datetime.fromtimestamp(obs_id)}")
    return obs_id


def remove_units(frame, units):
    "Remove units for g3 frame"
    if frame.type != core.G3FrameType.Map:
        return frame
    t_scale = units if frame['T'].weighted else 1./units
    w_scale = units * units
    for k in ['T', 'Q', 'U']:
        if k in frame:
            frame[k] = frame.pop(k) * t_scale
    for k in ['Wunpol', 'Wpol']:
        if k in frame:
            frame[k] = frame.pop(k) * w_scale
    return frame
