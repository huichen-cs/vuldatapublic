import json
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def load_m_list(json_fn):
    with open(json_fn, mode="rt") as f:
        m_list = json.load(f)
    return m_list


def hypothesis_test(m_list):
    good_list, bad_list = [], []
    for m in m_list:
        m_good = np.array(m["m_good"])
        m_bad = np.array(m["m_bad"])
        good_list.append(m_good)
        bad_list.append(m_bad)
    good = np.array(good_list)
    bad = np.array(bad_list)
    pvalues = []
    for i in range(1, good.shape[1] - 1):
        res = stats.ranksums(bad[:, i], good[:, i], "less")
        pvalues.append(res.pvalue)
        print(res)
    pvalue = stats.combine_pvalues(pvalues)
    print(pvalue)
    return pvalue


def main():
    data_dir = os.path.join("uncertainty", "result", "active")
    vcm_m_list = load_m_list(os.path.join(data_dir, "ps_vcmdata_active_learn.json"))
    sap_m_list = load_m_list(os.path.join(data_dir, "bert_sapdata_active_learn.json"))
    pvalue = hypothesis_test(vcm_m_list)
    print("VCM: ", pvalue)
    pvalue = hypothesis_test(sap_m_list)
    print("SAP: ", pvalue)


if __name__ == "__main__":
    main()
