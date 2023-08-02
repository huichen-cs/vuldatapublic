# Quantifying the Uncertainty in Software Vulnerability Patch Data

This repository contains the replication package and supplementary material for the titled research.

## Environment
The programs are written in Python and are tested in several Linux system
distributions. We recommend creating a Python virtual environment to run these
programs, such as
```bash
conda create -n vuldata python=3.9 pip
conda activate vuldata
pip install -r requirements.txt
```
The Python Requirements file is in the replication package.

## Directory Structures
- `data`: data
- `uncertainty/config`: experiment configuration files
- `uncertainty/src`: Python source code
- `uncertainty/scripts`: Shell scripts that run experiments by calling the Python programs

## Scripts and Program Naming Convention
The programs and scripts are written to experiment on combinations of UQ
approaches. These approaches combine data modeling (homoscedastic and heteroscedastic),
UQ estimation (vanilla, Model Ensemble, and Monte Carlo Dropout). We apply these
approaches to two datasets, SAP and VCMatch. For VCMatch dataset, we have two
types of data features, manually-crafted features (PatchScout features) and
embeddings (via CodeBert) while for SAP, we have embeddings. The shellscript
and the Python programs are generally named following the formula:
> ..._(data modeling)_(feature)_(data set)_...

For instance, `run_hmsc_bertsap_gn_shift_train.sh` is a shell script that run
experiment for UQ approach that train UQ model with homoscedastic data modeling
(`hmsc`) with SAP data set (`sap`) using CodeBert features (`bert`).

The script invokes Python program `hmsc_bert_sapdata_shift_train.py` to do the
training. The filename suggests that it train the UQ model with homoscedastic
data modeling (`hmsc`)  with SAP data set (`sapdata`) using CodeBert features
(`bert`).

To run the shell script, follow the help message given by the script:
```bash
run_hmsc_bertsap_gn_shift_train --help
```

To run the Python script, assume you are on the top directory of this repository,
we can retrieve the help message:
```bash
PYTHONPATH=uncertainty/src python uncertainty/src/hmsc_bert_sapdata_shift_train.py --help
```

## File Catalogue
|Purpose|Data Model|UQ Approximation|Feature| Dataset|Shell Script| Main Python Program | Python Package |
|-------|----------|----------------|-------|--------|------------|---------------------|----------------|
| RQ1: Dataset Quality Shift|Homoscedastic|Model Ensemble|PatchScout|VCMatch|run_hmsc_psvcm_gn_shift_train.sh|htsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Monte Carlo Dropout|PatchScout|VCMatch|run_hmsc_psvcm_gn_shift_train.sh|htsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Vanilla|PatchScout|VCMatch|run_hmsc_psvcm_gn_shift_train.sh|hmsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Model Ensemble|PatchScout|VCMatch|run_htsc_psvcm_gn_shift_train.sh|hmsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Monte Carlo Dropout|PatchScout|VCMatch|run_htsc_psvcm_gn_shift_train.sh|hmsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Vanilla|PatchScout|VCMatch|run_htsc_psvcm_gn_shift_train.sh|htsc_ps_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Model Ensemble|CodeBert|VCMatch|run_bert_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Monte Carlo Dropout|CodeBert|VCMatch|run_hmsc_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Vanilla|PatchScout|CodeBert|VCMatch|run_hmsc_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Model Ensemble|CodeBert|VCMatch|run_htsc_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Monte Carlo Dropout|CodeBert|VCMatch|run_htsc_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Vanilla|PatchScout|CodeBert|VCMatch|run_htsc_bertvcm_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Model Ensemble|CodeBert|SAP|run_bert_bertsap_gn_shift_train.sh|htsc_bert_sapdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Monte Carlo Dropout|CodeBert|SAP|run_hmsc_bertsap_gn_shift_train.sh|htsc_bert_vcmdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Homoscedastic|Vanilla|PatchScout|CodeBert|SAP|run_hmsc_bertsap_gn_shift_train.sh|htsc_bert_sapdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Model Ensemble|CodeBert|SAP|run_htsc_bertsap_gn_shift_train.sh|htsc_bert_sapdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Monte Carlo Dropout|CodeBert|SAP|run_htsc_bertsap_gn_shift_train.sh|htsc_bert_sapdata_shift_train.py|uqmodel.ensemble|
| RQ1: Dataset Quality Shift|Heteroscedastic|Vanilla|PatchScout|CodeBert|SAP|run_htsc_bertsap_gn_shift_train.sh|htsc_bert_sapdata_shift_train.py|uqmodel.ensemble|
