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
        help="Input file generated from active learning",
        default="output/linevul/commit/retrain/vcm_9_linevul_1_ehal_max.pt"
    )
    parser.add_argument(
        "--input_train_file",
        type=str,
        help="LineVul function-level train data file",
        default="data/linevul/splits/func_train.csv"
    )
    parser.add_argument(
        "--output_train_file",
        type=str,
        help="Output LineVul train data file",
        default="output/linevul/commit/retrain/selection/vcm_9_linevul_1_ehal_max.csv"
    )
    args = parser.parse_args()
    return args

def get_data_from_subset(data_subset):
    d = data_subset
    while (isinstance(d, torch.utils.data.dataset.Subset)
           and (not hasattr(d, 'data'))
           and hasattr(d, 'dataset')):
        d = d.dataset
    return d.data



def make_train_file():
    args = parse_command_line()
    #
    # load selected data
    #
    dataset_selected = torch.load(args.data_select_file)
    if not isinstance(dataset_selected, torch.utils.data.dataset.ConcatDataset):
        raise ValueError(
            "Expected the selected data concatenates two datasets, "
            f"but found {type(dataset_selected)=}"
        )
    if len(dataset_selected.datasets) < 2:
        raise ValueError(
            "Expected the selected dataset concatenates two datasets, "
            f"but found {len(dataset_selected.datasets)=}"
        )
    for dataset in dataset_selected.datasets[1:]:
        if not hasattr(dataset, "data"):
            raise ValueError(
                "Expected the selected dataset to have a data attribute, "
                f"but found {type(dataset)=}"
            )
        if not isinstance(dataset.data, pd.DataFrame):
            raise ValueError(
                "Expected the data attribute in selected dataset refers "
                "to a pandas DataFrame, "
                f"but found {type(dataset.data)=}"
            )
    selected_df = pd.concat([d.data for d in dataset_selected.datasets[1:]])
    selected_commits = selected_df["commit"]
    n_selected_commits = len(selected_commits)

    train_data = pd.read_csv(args.input_train_file)

    # offset the incorrect commit hash: 1ddf72180a52d247db88ea42a3e35f824a8fbda2
    #  that should have been 1ddf72180a52d247db88ea42a3e35f824a8fbda1
    # also there are also completely clean commits
    unique_commits=train_data[train_data["target"] == 1]["commit_id"].unique()
    n_train_commits = len(unique_commits)
    print(f"{n_selected_commits/n_train_commits=}")
    
    clean_only_df = train_data.drop(train_data[train_data["commit_id"].isin(unique_commits)].index)
    print(f"{len(clean_only_df)=}")
    selected_vul_df = train_data[train_data["commit_id"].isin(selected_commits)]
    print(f"{len(selected_vul_df)=}")
    
    new_train_df = pd.concat([selected_vul_df, clean_only_df])
    print(f"{len(new_train_df)/len(train_data)=}")
    new_train_df = new_train_df.sample(frac=1)
    new_train_df.to_csv(args.output_train_file, index=False)
    print(f"wrote {args.output_train_file}")

    data_df = pd.read_csv(args.output_train_file)
    wrote_commits = set(data_df["commit_id"])
    commits_set = set(selected_commits)
    print(f"{(wrote_commits == commits_set)=}")
    print(f"{len(data_df)/len(train_data)=}")

if __name__ == "__main__":
    make_train_file()
    print("done")
