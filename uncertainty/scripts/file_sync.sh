#!/bin/bash

# rsync -azvr --progress \
#         --exclude .git \
#         --exclude *.log \
#         --exclude *.log.* \
#         --exclude uq_testdata_ckpt \
#         -e 'ssh -p 52822' hchen@localhost:./work/vuldata/* vuldata/

# rsync -azvr --progress \
# 	methods \
# 	uncertainty \
# 	logger.ini \
# 	hui.chen@penzias:./uq/

rsync -azvr --progress \
	methods/VCMatch/data/Dataset_5000_*_shrinked.csv \
	hchen@192.168.1.118:/home/hchen/work/vuldata/methods/VCMatch/data/
rsync -azvr --progress \
	methods/VCMatch/data/repo_list.csv \
	hchen@192.168.1.118:/home/hchen/work/vuldata/methods/VCMatch/data/
rsync -azvr --progress \
	methods/VCMatch/data/dataset_cve_list.csv \
	hchen@192.168.1.118:/home/hchen/work/vuldata/methods/VCMatch/data/
