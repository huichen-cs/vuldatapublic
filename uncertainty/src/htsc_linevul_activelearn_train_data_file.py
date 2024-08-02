import argparse
import pandas as pd
import torch



def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [...]", description="Make LineVul train data file."
    )
    parser.add_argument(
        "--data_select_file",
        type=str,
        help="Input file generated from active learning"
    )
    parser.add_argument(
        "--input_train_file",
        type=str,
        help="Original LineVul train data file",
    )
    parser.add_argument(
        "--output_train_file",
        type=str,
        help="Output train data file for LineVul",
    )
    args = parser.parse_args()
    return args

def make_train_file():
    args = parse_command_line()
    #
    # load selected data
    #
    selected_data = torch.load(args.data_select_file)
    if len(selected_data.datasets) != 2:
        raise ValueError(
            "Expected the selected data concatenates two dataframes, "
            f"but found {len(selected_data.datasets)}"
        )
    subset = selected_data.datasets[1]
    selected_commits_df = subset.dataset.data.iloc[subset.indices]
    n_selected = len(selected_commits_df)
    selected_commits = selected_commits_df["commit"]
        
    train_data = pd.read_csv(args.input_train_file)

    # offset the incorrect commit hash: 1ddf72180a52d247db88ea42a3e35f824a8fbda2
    #  that should have been 1ddf72180a52d247db88ea42a3e35f824a8fbda1
    # also there are also completely clean commits
    unique_commits =train_data[train_data["target"] == 1]["commit_id"].unique()
    n_commits = len(unique_commits) - 1
    print(f"selected/n_commits = {n_selected}/{n_commits} = {n_selected/n_commits}")
    
    clean_only_df = train_data.drop(train_data[train_data["commit_id"].isin(unique_commits)].index)
    vul_clean_df = train_data[train_data["commit_id"].isin(selected_commits)]
    
    new_train_df = pd.concat([vul_clean_df, clean_only_df])
    print(len(new_train_df), len(new_train_df)/len(train_data))
    new_train_df = new_train_df.sample(frac=1)
    new_train_df.to_csv(args.output_train_file, index=False)
    print(f"wrote {args.output_train_file}")
   
if __name__ == "__main__":
    make_train_file()
