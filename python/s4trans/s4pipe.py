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

        # Read in catalog
        self.read_source_catalog()
        return

    def read_source_catalog(self):
        """ Read in the csv file with ra,dec,flux"""
        try:
            self.config.source_catalog
        except NameError:
            self.logger.warning("source_catalog doesn't exist")
            return

        if self.config.source_catalog is None:
            self.sources_coords = None
            return

        self.logger.info(f"Reading catalog with sources to insert: {self.config.source_catalog}")
        self.sources_coords = pd.read_csv(self.config.source_catalog)

        # Now we read the observation sequence
        self.obs_seq = s4tools.load_obs_seq()
        return

    def get_flux_scale(self, filename, obs_key, obs_width, scan, nsigma=2):

        filename = os.path.basename(filename)

        # Get the index for the file with the peak flux
        k = numpy.where(self.obs_seq[scan]['obs_seq'] == obs_key)[0][0]
        files = self.obs_seq[scan]['filename'][k-obs_width: k+obs_width+1].to_numpy()
        x = numpy.linspace(-nsigma*obs_width, nsigma*obs_width, 2*obs_width+1)
        g = s4tools.gaussian(x, obs_width)
        idx = numpy.where(files == filename)[0][0]
        scale = g[idx]
        return scale

    def get_flux_scales(self, filename, scan='RISING', nsigma=2):
        scales = []
        for row in self.sources_coords.itertuples():
            scale = self.get_flux_scale(filename, row.obs_key, row.obs_width, scan=row.scan, nsigma=nsigma)
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
        self.logger.info(f"Running spt3g_ingest version: {s4trans.__version__}")

    def load_healpix_map(self, filename, frame='T'):
        """Load a healpix map as an array"""
        t0 = time.time()
        if filename in self.hp_array:
            self.logger.info(f"Healpix map from: {filename} already stored -- skipping")
        else:
            self.logger.info(f"Reading into array file: {filename}")
            hp_array = maps.load_skymap_fits(filename)
            LOGGER.info(f"Read time: {s4tools.elapsed_time(t0)}")
            self.hp_array[filename] = hp_array[frame]
        return self.hp_array[filename]

    def write_fits(self, frame3g, file, proj_name):
        # Get self.outname and self.outname_tmp
        self.set_outname(file, proj_name, filetype='FITS')
        self.logger.info(f"Will create: {self.outname_tmp}")
        t0 = time.time()
        maps.fitsio.save_skymap_fits(self.outname_tmp, frame3g, overwrite=True)
        # In case we have indirect_write
        self.move_outname()
        self.logger.info(f"FITS file creation time: {s4tools.elapsed_time(t0)}")

    def write_g3(self, frame3g, file, proj_name):
        # Get self.outname and self.outname_tmp
        self.set_outname(file, proj_name, filetype='G3')
        self.logger.info(f"Will create: {self.outname_tmp}")
        t0 = time.time()
        f = core.G3Frame(core.G3FrameType.Map)
        f['T'] = frame3g
        f['T'].sparse = False
        core.G3Writer(filename=self.outname_tmp)(f)
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
            self.load_healpix_map(file)
            # Get the name of the output file to record the fractions
            self.set_outname_frac(file)
            ofile = open(self.outname_frac, 'w')

            k = 1
            for proj_name, proj in self.proj.items():

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

    def filter_sim_file(self, file, proj_name, band='150GHz'):
        """Filter Simulations using transient filtering method"""
        t0 = time.time()
        proj = self.proj[proj_name]
        self.load_healpix_map(file)
        t1 = time.time()
        self.logger.info(f"Transforming Healpix to G3 frame for projection {proj_name}")
        map3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")

        # Get the outname
        self.set_outname(file, f"flt_{proj_name}", filetype='G3')

        # Get the obs_id based on the name of the file
        obs_id = get_obs_id(file)

        # Insert if catalog is present
        if self.sources_coords is not None:
            ra = self.sources_coords['RA']
            dec = self.sources_coords['DEC']
            flux = self.sources_coords['FLUX']
            scale = self.get_flux_scales(file, scan='RISING', nsigma=2)
            flux_scaled = flux*scale
            map3g = insert_sources(map3g, ra, dec, flux_scaled, norm=False)

        # Create a weights maps of ones
        weights = map3g.clone()
        idx = numpy.where(weights != 0)
        numpy.asarray(weights)[idx] = 1
        weightmap = maps.G3SkyMapWeights()
        weightmap.TT = weights

        pipe = core.G3Pipeline()
        pipe.Add(core.G3InfiniteSource, n=0)
        pipe.Add(maps.InjectMaps, map_id=band, maps_in=[map3g, weightmap])

        pipe.Add(addid_to_frame, obs_id=obs_id)
        pipe.Add(maps.map_modules.MakeMapsUnpolarized)
        pipe.Add(transients.TransientMapFiltering,
                 bands=[band],  # or just band
                 subtract_coadd=False)
        pipe.Add(core.G3Writer, filename=self.outname_tmp)
        pipe.Run()

        # In case we have indirect_write
        self.move_outname()
        self.logger.info(f"Filtering file {file} done: {s4tools.elapsed_time(t0)} ")
        self.logger.info(f"Created file: {self.outname}")
        return

    def filter_sims(self):
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
            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")
            for proj_name in self.config.proj_name:
                on_fraction = self.get_db_onfraction(file, proj_name)
                if on_fraction > self.config.onfracion_thresh and on_fraction is not None:
                    self.filter_sim_file(file, proj_name)
                else:
                    self.logger.info(f"Skipping proj:{proj_name} for file: {file}")

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
                try:
                    on_fraction = self.get_db_onfraction(file, proj_name)
                except Exception:
                    self.logger.warning(f"Cannot find fraction for {file} on DB")
                    on_fraction = self.calculate_onfraction(file, proj_name)

                # Project if above threshold
                if on_fraction > self.config.onfracion_thresh and on_fraction is not None:
                    self.logger.info(f"Fraction of non-zero pixels above threshold: {on_fraction}")
                    self.project_sim_file(file, proj_name, filetypes)

                else:
                    self.logger.info(f"Fraction of non-zero pixels below threshold: {on_fraction}")
                    self.logger.info(f"Skipping FITS for projection: {proj_name}")

            self.logger.info(f"Done with {file} time: {s4tools.elapsed_time(t1)} ")
        nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")

    def project_sim_file(self, file, proj_name, filetypes):
        """ Project single sim file"""
        t0 = time.time()
        proj = self.proj[proj_name]
        self.load_healpix_map(file)
        t1 = time.time()
        self.logger.info(f"Transforming Healpix to G3 frame for projection {proj_name}")
        frame3g = maps.healpix_to_flatsky(self.hp_array[file], **proj)
        self.logger.info(f"Transforming done in: {s4tools.elapsed_time(t1)} ")
        self.logger.info(f"New Frame:\n {frame3g}")

        # Get the obs_id based on the name of the file
        # obs_id = get_obs_id(file)

        # Insert if catalog is present
        if self.sources_coords is not None:
            ra = self.sources_coords['RA']
            dec = self.sources_coords['DEC']
            flux = self.sources_coords['FLUX']
            scale = self.get_flux_scales(file, scan='RISING', nsigma=2)
            flux_scaled = flux*scale
            frame3g = insert_sources(frame3g, ra, dec, flux_scaled, norm=False)

        if 'FITS' in filetypes:
            self.write_fits(frame3g, file, proj_name)
        if 'G3' in filetypes:
            self.write_g3(frame3g, file, proj_name)
        self.logger.info(f"Projecting of file: {file} done: {s4tools.elapsed_time(t0)} ")
        return

    def ingest_onfraction(self):
        """Ingest the on fraction log files into the sqlite3 database"""

        t0 = time.time()
        # Make the connection to the DB
        con = s4tools.connect_db(self.config.dbname, self.config.tablename)

        # Loop over all of the files
        nfile = 1
        for filename in self.config.files:
            t1 = time.time()
            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")
            s4tools.ingest_fraction_file(filename, self.config.tablename, con=con)
            self.logger.info(f"Done with {filename} time: {s4tools.elapsed_time(t1)} ")
            nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")

    def get_db_onfraction(self, filename, proj_name):

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
            self.logger.info(f"Fraction: {fraction} for {filename}")
        except TypeError:
            fraction = None
            self.logger.warning(f"on-fraction not found for {filename} and {proj_name}")
        cur.close()
        return fraction


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


def insert_sources(frame, ra, dec, flux, norm=True, sigma=1.5, nsigma=2):

    """
    Inserts source at (ra,dec) using wcs information from frame
    If norm=True flux is spread over the kernel, otherwise it's the peak value
    at xo,yo. The code relies on the astropy.wcs object contained in the frame
    get the pixel coordinates.

    Arguments:
    ---------
    frame : 3g frame object
        The frame where we want to inject/insert point sourcers into.
    ra: float or list
        The list or float of R.A. in decimal degress
    decimal: float or list
        The list or float of Decl. in decimal degress
    flux: float or list
        The list or float with total flux of the source (units mJy)
    norm: Bool
        Normalize fluxe
    sigma: float
        The size of sigma
    nsigma: float
        The number of sigma to extend for the kernel
    """

    if not hasattr(ra, '__iter__') and not hasattr(dec, '__iter__') and not hasattr(flux, '__iter__'):
        ras = [ra]
        decs = [dec]
        fluxes = [flux]
    elif len(ra) == len(dec) and len(dec) == len(flux):
        ras = ra
        decs = dec
        fluxes = flux
    else:
        LOGGER.error("Error: ra, dec and flux must be same length")
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
    nobjects = len(ras)
    LOGGER.info(f"Will insert {nobjects} object(s) in 3G frame")
    for k in range(nobjects):
        ra = ras[k]
        dec = decs[k]

        # Transform the flux from mJy to K_CMB
        flux = s4tools.mJy2K_CMB(fluxes[k], freq=150, fwhm=1.0)

        # Generate the kernel with fluxes
        kernel = gaussian2d(flux, dim, dim, sigma, norm=norm)(y, x)

        # Get the pixel coordinates
        sky = SkyCoord(ra, dec, unit='deg')
        xo, yo = frame.wcs.world_to_pixel(sky)
        # Need to re-cast astropy/numpy objects as scalars -- and round
        xo = round(float(xo))
        yo = round(float(yo))
        # Now inject the kernel at xo,yo
        LOGGER.info(f"Inserting {dim,dim} kernel at {xo,yo} with flux {flux}")
        numpy.asarray(frame)[yo-dim:yo+dim+1, xo-dim:xo+dim+1] = kernel

    return frame


def addid_to_frame(fr, obs_id):
    """ Add obs_id to a frame"""
    fr['ObservationID'] = obs_id


def get_obs_id(file):
    """ Common method to get obs_id based on the name of the file"""
    date0 = datetime.datetime(2013, 1, 1, 0, 0).timestamp()
    f = os.path.basename(file)
    s = ''.join(f.split('_')[3].split('-'))
    obs_id = int(date0 + int.from_bytes(s.encode(), 'little')/1e5)
    LOGGER.info(f"Will add obs_id: {obs_id} to: {file}")
    return obs_id
