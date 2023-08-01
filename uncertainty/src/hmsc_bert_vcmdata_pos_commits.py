import argparse
import logging
import logging.config
import os

import pandas as pd
import tqdm

from typing import Tuple

from logconf.logging_utils import init_logging
from repodata.bert_data import (
    CodeBertPatchFile,
    PosIndexedPatchCollectionFileIterator,
    index_repo_patch_file,
    get_codebert_patch_filepath,
    get_repos,
    get_repo_cve,
    load_patch_file_index,
    make_repo_patch_log,
)

logger = logging.getLogger(__name__)


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


def get_cve_patches(
    repo_name: str,
    cve_df: pd.DataFrame,
    repo_patch_file: str,
    patch_index_file: str,
    overwrite=False,
):
    cve_codebert_file = get_codebert_patch_filepath(repo_patch_file, "pos")
    if not os.path.exists(cve_codebert_file) or overwrite:
        index_df = pd.read_csv(patch_index_file)
        cve_df = cve_df.rename(columns={"commit": "commit_hash"})
        indexed_cve_df = index_df.merge(cve_df, on="commit_hash")
        bertPatchFile = CodeBertPatchFile(cve_codebert_file)
        for cve_id, commit in PosIndexedPatchCollectionFileIterator(
            repo_name, repo_patch_file, indexed_cve_df
        ):
            logger.debug("cve: {}, commit: {}".format(cve_id, commit.commit_hash))
            bertPatchFile.add(cve_id, repo_name, commit)
        bertPatchFile.write_csv()
    else:
        logger.warning(
            "not to overwrite existing CodeBert Patch file {}".format(cve_codebert_file)
        )


def generate_pos_cve_patch_files(cve_patch_filepath: str, repos_dirpath: str):
    df = pd.read_csv(cve_patch_filepath)
    for repo_name in tqdm.tqdm(get_repos(df)):
        repo_patch_file = make_repo_patch_log(repos_dirpath, repo_name)
        patch_index_file = index_repo_patch_file(repo_patch_file)
        repo_df = get_repo_cve(df, repo_name)
        get_cve_patches(repo_name, repo_df, repo_patch_file, patch_index_file)
        logger.info("completed repo {}".format(repo_name))


if __name__ == "__main__":
    init_logging(__file__, append=True)
    cve_patch_filepath, repos_dirpath = get_cmdline_filepath()
    generate_pos_cve_patch_files(cve_patch_filepath, repos_dirpath)
