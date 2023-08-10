import numpy as np
import os
import pandas as pd


def split_train_val_test_indices(size: int, train_ratio, val_ratio):
    assert size > 0
    assert 0 < train_ratio + val_ratio <= 1

    pos_indices = list(range(size))
    np.random.shuffle(pos_indices)

    train_size = int(train_ratio * len(pos_indices))
    test_size = len(pos_indices) - train_size
    val_size = int(train_size * val_ratio)
    train_size = train_size - val_size
    assert (train_size + test_size + val_size) == len(pos_indices)
    train_indices, val_indices, test_indices = np.split(
        pos_indices, [train_size + 1, train_size + val_size + 1]
    )
    return train_indices, val_indices, test_indices


class SapData(object):
    POS_FILENAME = "SAP_full_commits.csv"
    NEG_FILENAME = "SAP_negative_commits_10x.csv"

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def train_test_val_split(self, train_ratio, test_ratio, val_ratio, neg_ratio):
        assert train_ratio + test_ratio == 1
        assert 0 < val_ratio < 1

        def func(x):
            return pd.concat([x[x["label"] == 1], x[x["label"] == 0].head(neg_ratio)])

        def load_pos_data():
            pos = pd.read_csv(
                os.path.join(self.data_dir, self.POS_FILENAME),
                index_col=0,
                keep_default_na=False,
            )
            pos["label"] = 1
            return pos

        def load_neg_data():
            neg = pd.read_csv(
                os.path.join(self.data_dir, self.NEG_FILENAME),
                index_col=0,
                keep_default_na=False,
            )
            neg["label"] = 0
            return neg

        def split_pos_neg_data(pos, neg, train_indices, val_indices, test_indices):
            pos_train = pos.iloc[train_indices]
            pos_val = pos.iloc[val_indices]
            pos_test = pos.iloc[test_indices]

            splits = {"train": None, "val": None, "test": None}
            for pos_split, split_name in zip(
                [pos_train, pos_val, pos_test], splits.keys()
            ):
                neg_split = neg[
                    neg["cve"].isin(pos_split["cve"])
                    & neg["commit"].isin(pos_split["commit"])
                ]
                pn_split = pd.concat([pos_split, neg_split], ignore_index=True)
                pn_split["cve_commit"] = pn_split["cve"] + "_" + pn_split["commit"]
                df_selected = pn_split.groupby("cve_commit").apply(func)
                df_selected = df_selected.reset_index(drop=True)
                splits[split_name] = df_selected
            return splits

        pos = load_pos_data()
        neg = load_neg_data()
        train_indices, val_indices, test_indices = split_train_val_test_indices(
            len(pos), train_ratio, val_ratio
        )
        splits = split_pos_neg_data(pos, neg, train_indices, val_indices, test_indices)

        return splits


if __name__ == "__main__":
    sap_data = SapData("methods/VCMatch/data/SAP")
    splits = sap_data.train_test_val_split(0.9, 0.1, 0.1, 2)
