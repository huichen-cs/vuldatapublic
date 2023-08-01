import argparse
import os

import pandas as pd
import tqdm

from typing import Tuple, Iterable

from repodata.bert_data import (
    get_codebert_patch_filepath,
    get_git_log_filepath,
    get_repos,
)


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


def combine_commit_files(repos_dirpath: str, repos: Iterable, file_type: str) -> None:
    df_list = []
    for repo_name in tqdm.tqdm(repos):
        log_filepath = get_git_log_filepath(repos_dirpath, repo_name)
        patch_filepath = get_codebert_patch_filepath(log_filepath, file_type)
        df = pd.read_csv(patch_filepath, encoding_errors="surrogateescape")
        df_list.append(df)
    df_all = pd.concat(df_list).reset_index()
    out_filepath = os.path.join(repos_dirpath, "ps_bert_{}.csv".format(file_type))
    df_all.to_csv(out_filepath, errors="surrogateescape")
    print("wrote " + out_filepath)


def combine_pos_neg_commits(cve_patch_filepath: str, repos_dirpath: str):
    df = pd.read_csv(cve_patch_filepath)
    repos = get_repos(df)
    combine_commit_files(repos_dirpath, repos, "pos")
    combine_commit_files(repos_dirpath, repos, "neg")


def main() -> None:
    cve_patch_filepath, repos_dirpath = get_cmdline_filepath()
    combine_pos_neg_commits(cve_patch_filepath, repos_dirpath)


if __name__ == "__main__":
    main()
