#!/bin/bash
set -x

jid1=$(sbatch j1_scan.sh | cut -f 4 -d' ')
jid2=$(sbatch --dependency=afterok:$jid1 j2_subset.sh | cut -f 4 -d' ')
jid3=$(sbatch --dependency=afterok:$jid2 j3_merge.sh | cut -f 4 -d' ')

squeue -u $USER -o "%.8A %.4C %.10m %.20E"

