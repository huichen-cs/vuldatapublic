import argparse
import functools
import glob
import logging
import logging.config
import operator
import os
import pandas as pd
import yaml
from typing import Tuple

LOG_CONFIG_FILEPATH = os.path.join("logconf", "logger_file.yml")
logger = logging.getLogger(__name__)


def init_logging(logfile, append=True):
    if os.path.exists(LOG_CONFIG_FILEPATH):
        logfilename = os.path.splitext(os.path.basename(logfile))[0] + ".log"
        if not append and os.path.exists(logfilename):
            logfilename = (
                os.path.splitext(os.path.basename(logfile))[0]
                + "_"
                + str(os.getpid())
                + ".log"
            )
        with open(LOG_CONFIG_FILEPATH, "rt") as f:
            logging_config = yaml.load(f, Loader=yaml.FullLoader)
        logging_config["handlers"]["file"]["filename"] = logfilename
        GLOBAL_LOG_FILEPATH = logfilename
        logging.config.dictConfig(logging_config)
    else:
        logging.warning(
            "log configuration file {} inaccessible, use basic configuration".format(
                LOG_CONFIG_FILEPATH
            )
        )
        logging.basicConfig(level=logging.INFO)


def get_cmdline_filepath() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="prepare PatchScout dataset for codebert classifier"
    )
    parser.add_argument(
        nargs=1,
        metavar="FilePath",
        help="CVE-ID and patch commit ID file",
        dest="cve_patch_file",
    )
    parser.add_argument(
        nargs=1,
        metavar="ReposDirPath",
        help="Path to the directory where repositories are",
        dest="repos_dir_path",
    )
    args = parser.parse_args()
    return args.cve_patch_file[0], args.repos_dir_path[0]


def count_pos_commits(repos_dir_path: str) -> int:
    def get_num_commits(filepath):
        df = pd.read_csv(filepath, encoding_errors="surrogateescape")
        return len(df)

    pos_csv_filepaths = os.path.join(repos_dir_path, "*_pos.csv")
    num_commits = functools.reduce(
        operator.add, map(get_num_commits, glob.glob(pos_csv_filepaths))
    )
    return num_commits


def count_cve_commits(cve_filepath: str) -> int:
    df = pd.read_csv(cve_filepath)
    return len(df)


if __name__ == "__main__":
    init_logging(__file__, append=True)
    cve_path_file, repos_dir_path = get_cmdline_filepath()
    pos_commits = count_pos_commits(repos_dir_path)
    cve_commits = count_cve_commits(cve_path_file)
    print(pos_commits, cve_commits)
