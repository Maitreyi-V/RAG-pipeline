#!/bin/bash
# Usage: bash run_kaggle.sh cv0            (first, alone)
#        bash run_kaggle.sh cv1 cv2 cv3 cv4 e3
set -e
for RUN in "$@"; do
  python build_examples.py --run $RUN
  python train.py --run $RUN --task verse
  python train.py --run $RUN --task off
  python score.py --run $RUN
  mkdir -p meta
  cp models/${RUN}_verse/train_meta.json meta/${RUN}_verse.json
  cp models/${RUN}_off/train_meta.json meta/${RUN}_off.json
  rm -rf models/${RUN}_*
done
python calibrate_eval.py
