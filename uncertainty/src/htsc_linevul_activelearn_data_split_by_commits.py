import argparse
import json
import os
import pandas as pd


def parse_commandline():
    parser = argparse.ArgumentParser("split data by commits")
    parser.add_argument(
        "input_data_file",
        help="Input commit patch data file",
        type=str
    )
    parser.add_argument(
        "split_ratio",
        help="Data split ratio, e.g, '[0.8, 0.1, 0.1]' a JSON array",
        type=str
    )
    parser.add_argument(
        "output_data_dir",
        help="Output data directory",
        type=str
    )
    parser.add_argument(
        "-p",
        "--output_file_prefix",
        help=(
            "Output file prefix, if no prefix given, "
            "the output files will be [train,val,test].csv"
        ),
        type=str
    )
    args = parser.parse_args()
    return args
    
def main(input_file, split_ratio, output_dir, file_prefix):
    in_df = pd.read_csv(input_file)
    commits = in_df["commit"]
    if len(commits) != len(commits.unique()):
        raise ValueError("Expected commit hashs are unqiue, found duplicates")
    
    if sum(split_ratio) != 1:
        raise ValueError(f"Sum of {split_ratio=} != 1")

    train_ratio, val_ratio, test_ratio = split_ratio
    
    train_df = in_df.sample(frac=train_ratio)
    remain_df = in_df.drop(train_df.index)
    if len(train_df) + len(remain_df) != len(in_df):
        raise ValueError(
            "incorrect split, expected "
            f"{(len(train_df) + len(remain_df))=} == {len(in_df)=}"
        )
    
    test_ratio = test_ratio/(val_ratio+test_ratio)
    test_df = remain_df.sample(frac=test_ratio)
    
    val_df = remain_df.drop(test_df.index)

    print(f"{len(train_df)=}, {len(val_df)=}, {len(test_df)=}")
    
    os.makedirs(output_dir)
    train_fn = os.path.join(output_dir, file_prefix + "train.csv")
    train_df.to_csv(train_fn, index=False)
    val_fn = os.path.join(output_dir, file_prefix + "val.csv")
    val_df.to_csv(val_fn, index=False)
    test_fn = os.path.join(output_dir, file_prefix + "test.csv")
    test_df.to_csv(test_fn, index=False)
    print(f"wrote {train_fn}, {val_fn}, {test_fn}")


if __name__ == "__main__":
    args = parse_commandline()
    main(
        args.input_data_file,
        json.loads(args.split_ratio),
        args.output_data_dir,
        file_prefix=args.output_file_prefix if args.output_file_prefix else ""
    )