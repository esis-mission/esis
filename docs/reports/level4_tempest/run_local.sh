#!/bin/bash
set -x
cd "$(dirname "$0")"
python -u local_mart.py --pitch 0.75 --num-velocity 6 --coadd  2>&1 | tee ~/esis_local_runs/log_coadd_0p75_nv6.txt
python -u local_mart.py --pitch 1.0  --num-velocity 12         2>&1 | tee ~/esis_local_runs/log_1p0_nv12.txt
python -u local_mart.py --pitch 0.75 --num-velocity 6          2>&1 | tee ~/esis_local_runs/log_0p75_nv6.txt
