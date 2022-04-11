from spt3g import core, maps
from s4trans import s4tools
import s4trans
import logging
import types
import time
import math
import os
from tempfile import mkdtemp
import shutil
import magic

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

        # Check input files vs file list
        self.check_input_files()

        # Define the projections
        self.proj = define_tiles_projection()
        return

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
        self.logger.info(f"Reading into array file: {filename}")
        hp_array = maps.load_skymap_fits(filename)
        LOGGER.info(f"Read time: {s4tools.elapsed_time(t0)}")
        self.hp_array = hp_array[frame]
        return self.hp_array

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
            for proj in self.proj:
                proj_name = f"proj{k}"
                self.logger.info(f"Transforming Healpix to G3 frame for projection: {proj_name}")
                t2 = time.time()
                frame3g = maps.healpix_to_flatsky(self.hp_array, **proj)
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

    def project_sims(self, filetypes=['FITS', 'G3']):

        t0 = time.time()

        # Make sure that the folder exists: n
        s4tools.create_dir(self.config.outdir)

        if self.config.indirect_write:
            self.tmpdir = mkdtemp(prefix=self.config.indirect_write_prefix)
            self.logger.info(f"Preparing folder for indirect_write: {self.tmpdir}")
            # Make sure that the folder exists:
            s4tools.create_dir(self.tmpdir)

        nfile = 1
        for file in self.config.files:

            self.logger.info(f"Doing: {nfile}/{self.config.nfiles} files")

            # Load up the gzip healpix map
            t1 = time.time()
            self.load_healpix_map(file)

            k = 1
            for proj in self.proj:
                proj_name = f"proj{k}"

                self.logger.info(f"Transforming Healpix to G3 frame for projection: {proj_name}")
                frame3g = maps.healpix_to_flatsky(self.hp_array, **proj)
                on_fraction = frame3g.npix_nonzero/frame3g.size

                if on_fraction > 0.05:
                    self.logger.info(f"Fraction of non-zero pixels above threshold: {on_fraction}")
                    self.logger.info(f"New Frame:\n {frame3g}")
                    if 'FITS' in filetypes:
                        self.write_fits(frame3g, file, proj_name)
                    if 'G3' in filetypes:
                        self.write_g3(frame3g, file, proj_name)
                else:
                    self.logger.info(f"Fraction of non-zero pixels below threshold: {on_fraction}")
                    self.logger.info(f"Skipping FITS for projection: {proj_name}")
                k += 1

            self.logger.info(f"Done with {file} time: {s4tools.elapsed_time(t1)} ")
        nfile += 1
        self.logger.info(f"Grand total time: {s4tools.elapsed_time(t0)} ")


def define_tiles_projection(ntiles=6, x_len=14000, y_len=20000,
                            res=7.27220521664304e-05,
                            weighted=False,
                            delta_center=-0.401834135):  # -23.024 in radians

    LOGGER.info(f"Will define {ntiles} tiles")
    proj = []
    dx = int(360./ntiles)
    for i in range(ntiles):
        LOGGER.info(f"Defining alpha_center for tile {i+1}  at {i*dx} degrees")
        alpha_center = i*dx*math.pi/180.
        p = {'res': res,
             'x_len': x_len,
             'y_len': y_len,
             'weighted': weighted,
             'alpha_center': alpha_center,
             'delta_center': delta_center,
             'proj': maps.MapProjection.ProjZEA}
        proj.append(p)
    return proj
