#!/bin/bash
# Stand-in for sbatch under `run.sh … --dry-run`: prints the command, submits nothing, returns a
# placeholder job id (so dependency chains print as they would be submitted).
N_FILE=${DRY_COUNTER:-/tmp/rfcav_dry_$USER}
n=$(( $(cat "$N_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$N_FILE"
out="[dry-run] sbatch"
for a in "$@"; do
  if [[ $a =~ ^[A-Za-z0-9_./:=,%@+-]+$ ]]; then out+=" $a"; else out+=" $(printf '%q' "$a")"; fi
done
echo "$out" >&2
echo "DRY$n"
