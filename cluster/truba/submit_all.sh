#!/bin/bash
# Submit every family of a group from families.tsv, each as its own array job.
#   cluster/truba/submit_all.sh            # group "train"
#   cluster/truba/submit_all.sh ood        # the out-of-distribution test families
#   cluster/truba/submit_all.sh all
set -euo pipefail
TRUBA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$TRUBA_DIR/config.sh"
GROUP=${1:-train}
for fam in $(family_list "$GROUP"); do
  "$TRUBA_DIR/submit_family.sh" "$fam"
done
