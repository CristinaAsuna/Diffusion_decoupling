#!/usr/bin/env bash
# Submit from the repository with your site's resource options, for example:
# sbatch --chdir="$PWD" --output=runs/slurm-%j.out --gres=gpu:1 jobs/train.slurm \
#   diffusion artifacts/prepared my-run --device cuda
# Create the runs directory before submission. Activate your Python environment first.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: sbatch [site options] jobs/train.slurm FAMILY PREPARED RUN_ID [train options]" >&2
  exit 2
fi
family="$1"
prepared="$2"
run_id="$3"
shift 3

exec decouple train --family "$family" --prepared "$prepared" --run-id "$run_id" "$@"
