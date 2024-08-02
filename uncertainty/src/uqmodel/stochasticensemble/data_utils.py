import ast
import json
import logging
import os
import pandas as pd


logger = logging.getLogger(__name__)


def load_repo_commits(commit_file: str) -> dict:
    if commit_file.endswith(".json"):
        with open(commit_file, encoding="utf-8", mode="r") as f:
            repo_commits = json.load(f)
    else:
        with open(commit_file, encoding="utf-8", mode="r") as f:
            repo_commits = ast.literal_eval(f.read())
    return repo_commits


def check_uniq_cve(patch_commit_df: pd.DataFrame) -> bool:
    valid = True
    for cve_group in patch_commit_df.groupby("cve"):
        cve_id, cve_df = cve_group[0], cve_group[1]
        if cve_df.repo.shape[0] > 1:
            logger.info(
                "CVE "
                + cve_id
                + " appears in "
                + " ".join(cve_df.repo)
                + ", unexpected more than once"
            )
            valid = False
        if cve_df.commit.shape[0] > 1:
            logger.info(
                "CVE " + cve_id + " has patch commits " + ", unexpected more t han one"
            )
            valid = False
    return valid


def get_repos_patch_commits_stats(
    repo_commits: dict, patch_commit_df: pd.DataFrame
) -> pd.DataFrame:
    stat_df = pd.DataFrame(
        columns=[
            "len(commits)",
            "len(set(commits))",
            "len(other+patch)",
            "len(patch_commits)",
            "len(set(patch_commits))",
            "ratio(unique)",
            "len(other_commits)",
        ]
    )
    for repo_group in patch_commit_df.groupby("repo"):
        repo_name, repo_df = repo_group[0], repo_group[1]
        commits = repo_commits[repo_name]
        patch_commits = repo_df.commit
        other_commits = set(commits) - set(patch_commits)
        row_df = pd.DataFrame(
            {
                "len(commits)": len(commits),
                "len(set(commits))": len(set(commits)),
                "len(other+patch)": len(other_commits) + len(set(patch_commits)),
                "len(patch_commits)": len(patch_commits),
                "len(set(patch_commits))": len(set(patch_commits)),
                "ratio(unique)": len(set(patch_commits)) / len(patch_commits),
                "len(other_commits)": len(other_commits),
            },
            index=[repo_name],
        )
        stat_df = pd.concat([stat_df, row_df])
        # stat_df = stat_df.append({
        #     'len(commits)': len(commits),
        #     'len(set(commits))': len(set(commits)),
        #     'len(other+patch)': len(other_commits) + len(set(patch_commits)),
        #     'len(patch_commits)': len(patch_commits),
        #     'len(set(patch_commits))': len(set(patch_commits)),
        #     'ratio(unique)': len(set(patch_commits))/len(patch_commits),
        #     'len(other_commits)': len(other_commits)
        #     }, ignore_index = True)
    return stat_df


def get_data_dup_patch_commits(patch_commit_df: pd.DataFrame) -> pd.DataFrame:
    duplicates_df_list = []
    for repo_group in patch_commit_df.groupby("repo"):
        repo_name, repo_df = repo_group[0], repo_group[1]
        patch_commits = repo_df.commit
        seen = set()
        duplicates = [x for x in patch_commits if x in seen or seen.add(x)]
        duplicates_repo_df = repo_df.loc[repo_df.commit.isin(duplicates)]
        if duplicates:
            logger.info("Project " + repo_name + " has duplicated patch commits")
        duplicates_df_list.append(duplicates_repo_df)
    duplicates_patch_commit_df = pd.concat(duplicates_df_list)
    return duplicates_patch_commit_df


def get_repo_feature_filename(data_dir, file_prefix, repo_name, use_shrinked=True):
    if use_shrinked:
        return os.path.join(data_dir, file_prefix + "_" + repo_name + "_shrinked.csv")
    else:
        return os.path.join(data_dir, file_prefix + "_" + repo_name + ".csv")


def load_dataset_feature_data(data_dir, file_prefix, repo_list):
    df_list = []
    for repo_name in repo_list:
        repo_fn = get_repo_feature_filename(data_dir, file_prefix, repo_name)
        logger.debug("loading " + repo_fn + " ... ")
        repo_df = pd.read_csv(repo_fn)
        df_list.append(repo_df)
    df = pd.concat(df_list, ignore_index=True)
    return df


def get_repo_list(data_dir, repo_list_fn="repo_list.csv"):
    fn = os.path.join(data_dir, repo_list_fn)
    df = pd.read_csv(fn)
    return df["Repository"].to_list()


def get_dataset_cve_list(data_dir, cve_list_fn="dataset_cve_list.csv"):
    fn = os.path.join(data_dir, cve_list_fn)
    df = pd.read_csv(fn)
    return df["cve"].unique()


def get_patchscout_feature_list():
    # patch scout features
    ps_columns = [
        # (VID) Vulnerability Identifier
        #     CVE-ID                                     Whether the code commit mentions the CVE-ID of the target vulnerability.
        #     Software-specific Bug-ID.	               Whether the code commit mentions the software-specific Bug-ID in the NVD
        "cve_match",
        "bug_match",
        # (VL) Vulnerability Location
        #     Same File Num 	                           # of files that appear in both code commit and NVD description.
        #     Same File Ratio                            # of same files / # of files that appear in the NVD description.
        #     Unrelated File Num                         # of files that appear in code commit but not mentioned in the NVD description.
        #     Same Function Num                          # of functions that appear in both code commit and NVD description.
        #     Same Function Ratio                        # of same functions / # of functions that appear in the NVD description.
        #     Unrelated Function Num description.        # of functions that appear in code commit but not mentioned in the NVD
        "file_same_cnt",
        "file_same_ratio",
        "file_unrelated_cnt",
        "func_same_cnt",
        "func_same_ratio",
        "func_unrelated_cnt",
        # 'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',  # VCMatch
        # (VT) Vulnerability Type
        #     Vulnerability Type Relevance               The relevance of the vulnerability type-related texts between NVD information and commit message.
        #         3-tuple:
        #             Inclusion relationship
        #             Causality relationship
        #             Irrelevance relationship
        #     Patch Likelihood                           The probability of a commit to be a security patch.
        "vuln_type_1",
        "vuln_type_2",
        "vuln_type_3",  # 'patchlike', # already in data_df
        # (VDT) Vulnerability Descriptive Texts
        #     Shared-Vul-Msg-Word Num	                   # of shared words between NVD description and commit message.
        #     Shared-Vul-Msg-Word Ratio	               # of Shared-Vul-Msg-Words / # of words in NVD description.
        #     Max of Shared-Vul-Msg-Word Frequency 	   The max of the frequencies for all Shared-Vul-Msg-Words.
        #     Sum of Shared-Vul-Msg-Word Frequency 	   The sum of the frequencies for all Shared-Vul-Msg-Words.
        #     Average of Shared-Vul-Msg-Word Frequency   The average of the frequencies for all Shared-Vul-Msg-Words.
        #     Variance of Shared-Vul-Msg-Word Frequency  The variance of the frequencies for all Shared-Vul-Msg-Words.
        #     Shared-Vul-Code-Word Num	               # of shared words between NVD description and code diff.
        #     Shared-Vul-Code-Word Ratio 	               # of Shared-Vul-Code-Words / # of words in NVD description.
        #     Max of Shared-Vul-Code-Word Frequency 	   The max of the frequencies for all Shared-Vul-Code-Words.
        #     Sum of Shared-Vul-Code-Word Frequency 	   The sum of the frequencies for all Shared-Vul-Code-Words.
        #     Average of Shared-Vul-Code-Word Frequency  The average of the frequencies for all Shared-Vul-Code-Words.
        #     Variance of Shared-Vul-Code-Word Frequency The variance of the frequencies for all Shared-Vul-Code-Words.
        "mess_shared_num",
        "mess_shared_ratio",
        "mess_max",
        "mess_sum",
        "mess_mean",
        "mess_var",
        "code_shared_num",
        "code_shared_ratio",
        "code_max",
        "code_sum",
        "code_mean",
        "code_var",
    ]
    return ps_columns
