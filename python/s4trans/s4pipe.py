from spt3g import maps
from s4trans import s4tools
import logging
import types
import time
import math
import os

LOGGER = logging.getLogger(__name__)


class S4pipe:

    """ A Class to run and manage the Header Service"""

    def __init__(self, **keys):
        self.config = types.SimpleNamespace(**keys)

        self.setup_logging()
        self.proj = define_tiles_projection()

        return

    def setup_logging(self):
        """ Simple logger that uses configure_logger() """
        # Create the logger
        s4tools.create_logger(level=self.config.loglevel,
                              log_format=self.config.log_format,
                              log_format_date=self.config.log_format_date)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging Started at level:{self.config.loglevel}")

    def project_sims(self):
        for file in self.config.files:
            # Load up the gzip healpix map
            hp_array = maps.load_skymap_fits(file)

            # The basename for the output
            basename = file.split('.')[0]
            k = 1
            for proj in self.proj:
                proj_name = f"proj{k}"
                self.logger.info("Transforming Healpix to frame")
                frame3g = maps.healpix_to_flatsky(hp_array['T'], **proj)
                self.logger.info(f"New Frame:\n {frame3g}")
                fitsfile = f"{basename}_{proj_name}.fits"
                self.logger.info(f"Will create: {fitsfile}")
                t0 = time.time()
                maps.fitsio.save_skymap_fits(fitsfile, frame3g, overwrite=True)
                self.logger.info(f"FITS file creation time: {s4tools.elapsed_time(t0)}")
                k = k+1

            self.logger.info(f"Done with {file}")


def load_healpix_sim(filename, frame='T'):
    "Load a CMBS4 DC1 simulation in healpix format"
    t0 = time.time()
    LOGGER.info(f"Reading into array file: {filename}")
    hp_array = maps.load_skymap_fits(filename)
    LOGGER.info(f"Read time: {s4tools.elapsed_time(t0)}")
    return hp_array[frame]


def define_tiles_projection(ntiles=6, x_len=14000, y_len=20000,
                            res=7.27220521664304e-05,
                            weighted=False,
                            delta_center=-0.401834135):  # -23.024 in radians

    LOGGER.info(f"Will define {ntiles} tiles")

    proj = []
    dx = int(360./ntiles)
    for i in range(ntiles):
        LOGGER.info(f"Defining alpha_center for tile {i}  at {i*dx} degrees")
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
