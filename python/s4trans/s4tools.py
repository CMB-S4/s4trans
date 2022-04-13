
import time
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import errno
import sqlite3
from s4trans.data_types import Fd


LOGGER = logging.getLogger(__name__)

# SQL string definitions
# Create SQL statement to create table automatically
_table_statement = ''
for k in Fd.keys():
    _table_statement += '{} {},\n'.format(k, Fd[k])
# remove last comma
_table_statement += 'UNIQUE(ID) '
_table_statement = _table_statement.rstrip(',\n')

# Template to insert a row
_insert_row = """
INSERT{or_replace}INTO {tablename} values ({values})
"""


def create_logger(logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Simple logger that uses configure_logger()
    """
    logger = logging.getLogger(__name__)
    configure_logger(logger, logfile=logfile, level=level,
                     log_format=log_format, log_format_date=log_format_date)
    logging.basicConfig(handlers=logger.handlers, level=level)
    logger.propagate = False
    return logger


def configure_logger(logger, logfile=None, level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Configure an existing logger
    """
    # Define formats
    if log_format:
        FORMAT = log_format
    else:
        FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    if log_format_date:
        FORMAT_DATE = log_format_date
    else:
        FORMAT_DATE = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(FORMAT, FORMAT_DATE)

    # Need to set the root logging level as setting the level for each of the
    # handlers won't be recognized unless the root level is set at the desired
    # appropriate logging level. For example, if we set the root logger to
    # INFO, and all handlers to DEBUG, we won't receive DEBUG messages on
    # handlers.
    logger.setLevel(level)

    handlers = []
    # Set the logfile handle if required
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=2000000, backupCount=10)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
        logger.addHandler(fh)

    # Set the screen handle
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    handlers.append(sh)
    logger.addHandler(sh)
    return


def create_dir(dirname):
    "Safely attempt to create a folder"
    if not os.path.isdir(dirname):
        LOGGER.info(f"Creating directory {dirname}")
        try:
            os.makedirs(dirname, mode=0o755, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                LOGGER.warning(f"Problem creating {dirname} -- proceeding with trepidation")


def elapsed_time(t1, verb=False):
    """
    Returns the time between t1 and the current time now
    I can can also print the formatted elapsed time.
    ----------
    t1: float
        The initial time (in seconds)
    verb: bool, optional
        Optionally print the formatted elapsed time
    returns
    -------
    stime: float
        The elapsed time in seconds since t1
    """
    t2 = time.time()
    stime = "%dm %2.2fs" % (int((t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print("Elapsed time: {}".format(stime))
    return stime


def connect_db(dbname, tablename):
    """Establisih connection to DB"""

    LOGGER.info(f"Establishing DB connection to: {dbname}")

    # Connect to DB
    # SQLlite DB lives in a file
    con = sqlite3.connect(dbname)

    # Create the table
    create_table = """
    CREATE TABLE IF NOT EXISTS {tablename} (
    {statement}
    )
    """.format(**{'tablename': tablename, 'statement': _table_statement})
    LOGGER.debug(create_table)

    cur = con.cursor()
    cur.execute(create_table)
    con.commit()
    return con


def check_dbtable(dbname, tablename):
    """ Check tablename exists in database"""
    LOGGER.info(f"Checking {tablename} exits in: {dbname}")
    # Connect to DB
    con = sqlite3.connect(dbname)
    # Create the table
    create_table = """
    CREATE TABLE IF NOT EXISTS {tablename} (
    {statement}
    )
    """.format(**{'tablename': tablename, 'statement': _table_statement})
    LOGGER.debug(create_table)

    cur = con.cursor()
    cur.execute(create_table)
    con.commit()
    con.close()
    return


def ingest_fraction_file(filename, tablename, con=None, dbname=None, replace=False):
    """Ingest fractions from files into sqlite3 database"""

    # Make new connection if not available
    if not con:
        close_con = True
        con = sqlite3.connect(dbname)
    else:
        close_con = False

    # Get cursor
    cur = con.cursor()

    # Replace or not
    if replace:
        or_replace = ' OR REPLACE '
    else:
        or_replace = ' '

    # Read in the file
    LOGGER.info(f"Ingesting: {filename} to: {tablename}")
    with open(filename) as file:
        for line in file:
            SIMID, PROJ, FRACTION = line.rstrip().split()
            ID = f"{SIMID}_{PROJ}"

            # Create the ingested values in the same order,
            values = [ID, SIMID, PROJ, FRACTION]
            # Convert the values into a long string
            values_str = ", ".join(f'"{x}"' for x in values)

            # Format the insert query
            insert_query = _insert_row.format(**{'or_replace': or_replace,
                                                 'tablename': tablename, 'values': values_str})
            LOGGER.debug(f"Executing:{insert_query}")
            try:
                cur.execute(insert_query)
                con.commit()
                LOGGER.info(f"Ingestion Done for ID: {ID}")
            except sqlite3.IntegrityError:
                LOGGER.warning(f"NOT UNIQUE: ingestion failed for {ID}")

    # Done close connection
    if close_con:
        con.close()
    return
