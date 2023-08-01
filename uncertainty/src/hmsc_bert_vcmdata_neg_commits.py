import argparse
import logging
import logging.config
import os

import numpy as np
import pandas as pd
import tqdm

from typing import Tuple

from logconf.logging_utils import init_logging
from repodata.bert_data import (
    CodeBertPatchFile,
    NegIndexedPatchCollectionFileIterator,
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
    parser.add_argument(
        nargs=1,
        metavar="NegToPosRatio",
        help="Desired ratio of negative to postive commits ratio",
        type=int,
        dest="neg_to_pos_ratio",
    )
    parser.add_argument(
        nargs="?",
        metavar="ExtraSkipFactor",
        help="Ratio of extra commits in case of parsing errors",
        type=float,
        dest="extra_skip_factor",
        default=1.0,
    )
    args = parser.parse_args()
    return (
        args.cve_patch_file[0],
        args.repos_dir_path[0],
        args.neg_to_pos_ratio[0],
        args.extra_skip_factor,
    )


def sample_non_cve_patches(
    index_df: pd.DataFrame,
    cve_df: pd.DataFrame,
    sampling_factor: int = 10,
    extra_skip_factor: float = 1,
) -> Tuple[pd.DataFrame, int]:
    # 1. rename commit to commit_hash in order to do a DataFrame merge
    cve_df = cve_df.rename(columns={"commit": "commit_hash"})
    # 2. obtain common commits via an inner join
    common_df = index_df.merge(cve_df, on="commit_hash")
    # 3. obtain non-cve-patch commits via an outer join
    merged = index_df.merge(cve_df, on="commit_hash", how="outer", indicator=True)
    none_df = merged[merged["_merge"] == "left_only"]
    indexed_non_cve_df = none_df.drop(columns=["_merge"])
    # 4. determine the number of samples to draw based on sampling factor,
    #    the non-cve-patch to cve-patch ratio, and obtain the samples
    num_wanted = len(common_df) * sampling_factor
    num_extra = int(np.round(num_wanted * extra_skip_factor))
    num_actual = len(indexed_non_cve_df)
    if num_wanted <= num_actual:
        sampled_df = indexed_non_cve_df.sample(n=num_wanted, ignore_index=True)
        extra_df = indexed_non_cve_df.sample(n=num_extra, ignore_index=True)
        wanted_df = pd.concat([sampled_df, extra_df]).sample(frac=1, ignore_index=True)
    else:
        n_samples = num_wanted - num_actual + num_extra
        samples_df = indexed_non_cve_df.sample(n=n_samples, replace=True)
        wanted_df = pd.concat([indexed_non_cve_df, samples_df]).sample(
            frac=1, ignore_index=True
        )
    # 5. duplicate common-commit rows in order to fill the missing cve, repo, and cve-patch commit hashes
    expanded_common_df = common_df.loc[
        common_df.index.repeat(sampling_factor)
    ].reset_index(drop=True)
    # 6. preserve actual commit hash for the non-cve-patch commits
    wanted_df = wanted_df.rename(columns={"commit_hash": "neg_commit_hash"})
    # 7. fill the cve, repo, and cve-patch commit hashes
    wanted_df[["ref_pos_commit_hash", "ref_cve", "repo"]] = expanded_common_df[
        ["commit_hash", "cve", "repo"]
    ]
    # 8. correct undesired data type introduced by merge
    wanted_df[["commit_offset", "n_lines"]] = wanted_df[
        ["commit_offset", "n_lines"]
    ].astype(int)
    # 9. drop 'cve' to avoid confusion
    wanted_df = wanted_df.drop(columns=["cve"])
    return wanted_df, num_wanted


def get_non_cve_patches(
    repo_name: str,
    cve_df: pd.DataFrame,
    repo_patch_file: str,
    patch_index_file: str,
    overwrite: bool = False,
    sampling_factor: int = 10,
    extra_skip_factor: float = 1,
):
    cve_codebert_file = get_codebert_patch_filepath(repo_patch_file, "neg")
    if not os.path.exists(cve_codebert_file) or overwrite:
        index_df = load_patch_file_index(patch_index_file)
        indexed_non_cve_df, num_wanted = sample_non_cve_patches(
            index_df,
            cve_df,
            sampling_factor=sampling_factor,
            extra_skip_factor=extra_skip_factor,
        )
        bertPatchFile = CodeBertPatchFile(cve_codebert_file, commit_type="neg")
        for i, (ref_cve_id, neg_commit, ref_pos_commit_hash) in enumerate(
            NegIndexedPatchCollectionFileIterator(
                repo_name, repo_patch_file, indexed_non_cve_df
            )
        ):
            if i >= num_wanted:
                break
            logger.debug(
                "cve: {}, neg_commit: {}, ref_pos_commit: {}".format(
                    ref_cve_id, neg_commit.commit_hash, ref_pos_commit_hash
                )
            )
            bertPatchFile.add(ref_cve_id, repo_name, neg_commit, ref_pos_commit_hash)
        bertPatchFile.write_csv()
    else:
        logger.warning(
            "not to overwrite existing CodeBert Patch file {}".format(cve_codebert_file)
        )


def generate_neg_cve_patch_files(
    cve_patch_filepath: str,
    repos_dirpath: str,
    sampling_factor: int = 10,
    extra_skip_factor: float = 1,
):
    df = pd.read_csv(cve_patch_filepath)
    for repo_name in tqdm.tqdm(get_repos(df)):
        # if repo_name != 'linux':
        #     continue
        repo_patch_file = make_repo_patch_log(repos_dirpath, repo_name)
        patch_index_file = index_repo_patch_file(repo_patch_file)
        repo_df = get_repo_cve(df, repo_name)
        get_non_cve_patches(
            repo_name,
            repo_df,
            repo_patch_file,
            patch_index_file,
            sampling_factor=sampling_factor,
            extra_skip_factor=extra_skip_factor,
        )
        logger.info("completed repo {}".format(repo_name))


if __name__ == "__main__":
    init_logging(
        __file__, append=True, config_file=os.path.join("logconf", "logger_file.yml")
    )
    (
        cve_patch_filepath,
        repos_dirpath,
        sampling_factor,
        extra_skip_factor,
    ) = get_cmdline_filepath()
    generate_neg_cve_patch_files(cve_patch_filepath, repos_dirpath, sampling_factor)
