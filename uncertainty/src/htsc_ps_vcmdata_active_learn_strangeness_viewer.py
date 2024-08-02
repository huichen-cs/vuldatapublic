import json
import logging
import os
import pickle
from uqmodel.stochasticensemble.active_learn import EnsembleCheckpoint


logger = logging.getLogger(__name__)


def load_from_checkpoint(ckpt_path):
    ckpt = EnsembleCheckpoint(
        5,
        ckpt_path,
        warmup_epochs=0,
        tag=None,
        train=False,
    )
    try:
        _, _, test_dataset, ps_columns = ckpt.load_datasets()
        logger.info(
            "loaded train/val/test datasets from checkpoint at {}".format(ckpt_path)
        )
    except FileNotFoundError as err:
        logger.error(
            "failed to load train/va/test datasets from checkpoint at {}".format(
                ckpt_path
            )
        )
        raise err
    return test_dataset, ps_columns


def compute_goodness(json_file):
    uq_data = json.load(open(json_file, "r"))
    for method in uq_data.keys():
        if not (method == "ehal" and len(uq_data[method]) > 0):
            continue
        goodness = [
            epi / ale
            for epi, ale in zip(
                uq_data[method]["entropy_epistermic"],
                uq_data[method]["entropy_aleatoric"],
            )
        ]
    if goodness:
        return goodness
    else:
        raise ValueError("No goodness computed")


def count_type_in_halves(dataset, goodness):
    print(len(goodness))
    goodness_index = sorted(range(len(goodness)), key=lambda k: goodness[k])
    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) // 2)
    ]
    print(sum(match_list))

    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) // 2, len(goodness_index))
    ]
    print(sum(match_list))


def count_type_in_quarters(dataset, goodness):
    print(len(goodness))
    goodness_index = sorted(range(len(goodness)), key=lambda k: goodness[k])
    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) // 4)
    ]
    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) // 4)
    ]
    print(sum(match_list))

    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) // 4, len(goodness_index) * 2 // 4)
    ]
    print(sum(match_list))

    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) * 2 // 4, len(goodness_index) * 3 // 4)
    ]
    print(sum(match_list))

    match_list = [
        dataset.df.iloc[goodness_index[i]].true_commit
        == dataset.df.iloc[goodness_index[i]].commit
        for i in range(len(goodness_index) * 3 // 4, len(goodness_index) * 4 // 4)
    ]
    print(sum(match_list))


def find_bad_good_changeset(dataset, goodness):
    goodness_index = sorted(range(len(goodness)), key=lambda k: goodness[k])
    index_list = [
        i
        for i in goodness_index
        if dataset.df.iloc[i].true_commit == dataset.df.iloc[i].commit
    ]
    for i in range(10):
        print(
            dataset.df.iloc[index_list[i]]["cve"],
            dataset.df.iloc[index_list[i]]["repo"],
            dataset.df.iloc[index_list[i]]["commit"],
            dataset.df.iloc[index_list[i]]["vuln_type_1"],
            dataset.df.iloc[index_list[i]]["vuln_type_2"],
            dataset.df.iloc[index_list[i]]["vuln_type_3"],
        )
    print("4------------------------------")
    for i in range(len(index_list) - 10, len(index_list)):
        print(
            dataset.df.iloc[index_list[i]]["cve"],
            dataset.df.iloc[index_list[i]]["repo"],
            dataset.df.iloc[index_list[i]]["commit"],
            dataset.df.iloc[index_list[i]]["vuln_type_1"],
            dataset.df.iloc[index_list[i]]["vuln_type_2"],
            dataset.df.iloc[index_list[i]]["vuln_type_3"],
        )


def main(json_file, ckpt_path):
    dataset, _ = load_from_checkpoint(ckpt_path)

    goodness_file = "goodness.pkl"
    if not os.path.exists(goodness_file):
        goodness = compute_goodness(json_file)
        pickle.dump(goodness, open(goodness_file, "wb"))
    else:
        goodness = pickle.load(open(goodness_file, "rb"))
    print("1-----------------")
    count_type_in_halves(dataset, goodness)
    print("2-----------------")
    count_type_in_quarters(dataset, goodness)
    print("3-----------------")
    find_bad_good_changeset(dataset, goodness)


if __name__ == "__main__":
    ckpt_path = "uq_testdata_ckpt/htscpsvcm/active/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0/"
    in_json_file = "uncertainty/result/htscpsvcm/active/analysis/en_psvcm_1.0_im_1_gamma_0.5_train_0.9_sigma_0.json"
    main(in_json_file, ckpt_path)
