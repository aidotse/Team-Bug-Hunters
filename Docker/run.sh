#!/bin/bash
# File              : run.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 23.10.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

ROOT_DIR=$PWD/..
DATA_DIR=/mnt/astra_data
CODE_DIR=$ROOT_DIR/src

nvidia-docker  run   \
	-p 8888:8888 \
	-e JUPYTER_TOKEN="acic_bughunters" \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/src \
	-it bug_hunters_docker_image_tf2 \
	bash 

