#!/bin/bash
# Cleanup all .ipynb_checkpoints littering the leavesdb-v1.0 dataset on disk.

export IMAGE_ROOT="/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v1_0/images"

echo "Cleaning recursively: $IMAGE_ROOT"

rm -rvf `find $IMAGE_ROOT -type d -name .ipynb_checkpoints`

echo "Finished cleaning"
