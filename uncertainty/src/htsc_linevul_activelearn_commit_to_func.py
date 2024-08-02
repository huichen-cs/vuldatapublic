import argparse
import pandas as pd

LINEVUL_CSV_DATA_TYPES = {
    "Access Gained": str,
    "commit_message": str,
    "file_name": str,
    "files_changed": str,
    "lines_after": str,
    "lines_before": str,
    "parentID": str,
    "flaw_line": str,
    "flaw_line_index": str,
}


def fix_incorrect_commit_hash(df, inplace=True):
    # for commit 1ddf72180a52d247db88ea42a3e35f824a8fbda2, drop it because the correct one
    # 1ddf72180a52d247db88ea42a3e35f824a8fbda1 already in the dataframe
    df.replace(
        '1ddf72180a52d247db88ea42a3e35f824a8fbda2',
        '1ddf72180a52d247db88ea42a3e35f824a8fbda1',
        inplace=inplace)
    return df

def parse_commandline():
    parser = argparse.ArgumentParser(
        "make functional level file from commit file"
    )
    parser.add_argument("commit_file",
               help="commit level input data file",
               type=str)
    parser.add_argument("func_file",
               help="function level input data file",
               type=str)
    parser.add_argument("output_file",
               help="funtion level output data file",
               type=str)
    args = parser.parse_args()
    return args

def main(commit_fn, func_fn, output_fn):
    commit_df = pd.read_csv(commit_fn)
    commits = commit_df["commit"]
    func_df = pd.read_csv(func_fn, dtype=LINEVUL_CSV_DATA_TYPES)
    fix_incorrect_commit_hash(func_df, inplace=True)
    selected_df = func_df[func_df["commit_id"].isin(commits)]
    print(f"{len(selected_df)=}")
    selected_df.to_csv(output_fn, index=False)
    print(f"wrote {output_fn}")

    
    

if __name__ == "__main__":
    args = parse_commandline()
    main(args.commit_file, args.func_file, args.output_file)
