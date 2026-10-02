#!/bin/bash
# Submit every family of a group from families.tsv, each as its own array + conversion.
#   cluster/truba/submit_all.sh            # group "train"
#   cluster/truba/submit_all.sh ood        # the out-of-distribution test families
#   cluster/truba/submit_all.sh all [--no-convert]
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
GROUP=${1:-train}
shift || true
for fam in $(awk -v g="$GROUP" '$1 !~ /^#/ && NF >= 5 && (g == "all" || $5 == g) {print $1}' "$FAMILIES_TSV"); do
  "$TRUBA_DIR/submit_family.sh" "$fam" "$@"
done
