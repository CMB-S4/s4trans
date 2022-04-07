#!/usr/bin/env python

from s4trans import s4pipe
import argparse
import time


def cmdline():

    parser = argparse.ArgumentParser(description="CMBS4 transient tests pipeline")
    parser.add_argument("files", nargs='+',
                        help="Filename(s) to preocess")

    # Logging options (loglevel/log_format/log_format_date)
    default_log_format = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    default_log_format_date = '%Y-%m-%d %H:%M:%S'
    parser.add_argument("--loglevel", action="store", default='INFO', type=str.upper,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging Level [DEBUG/INFO/WARNING/ERROR/CRITICAL]")
    parser.add_argument("--log_format", action="store", type=str, default=default_log_format,
                        help="Format for logging")
    parser.add_argument("--log_format_date", action="store", type=str, default=default_log_format_date,
                        help="Format for date section of logging")
    args = parser.parse_args()

    # The total number of files
    args.nfiles = len(args.files)

    return args


if __name__ == "__main__":

    # Keep time
    t0 = time.time()

    # Get the command-line arguments
    args = cmdline()
    p = s4pipe.S4pipe(**args.__dict__)
    p.project_sims()
