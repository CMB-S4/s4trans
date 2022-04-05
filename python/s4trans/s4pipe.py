
from spt3g import maps
from s4trans import s4tools
import logging
import types

LOGGER = logging.getLogger(__name__)

class S4pipe:

    """ A Class to run and manage the Header Service"""

    def __init__(self, **keys):
        self.config = types.SimpleNamespace(**keys)

        self.setup_logging()
        return

    def setup_logging(self):
        """ Simple logger that uses configure_logger() """
        # Create the logger
        s4tools.create_logger(level=self.config.loglevel,
                              log_format=self.config.log_format,
                              log_format_date=self.config.log_format_date)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging Started at level:{self.config.loglevel}")


def load_healpix_sim(filename, frame='T'):
    "Load a CMBS4 DC1 simulation in healpix format"
    LOGGER.info(f"-- Reading into array file: {filename}")
    hp_array = maps.load_skymap_fits(filename)
    return hp_array[frame]
