# source env.sh   (from anywhere; bash or zsh)
# Puts the three source roots of this repository on PYTHONPATH:
#   real2sim/src            Part 1  packages real2sim_harness (agent harness) and calibrated_scene_reconstruction (metric scene pipeline)
#   policy_development/src  Part 2  package astra_station (policy-development controller)
#   end_to_end/src          chain   package astra_e2e (oversight, unified agent tools, the end-to-end chain)
# Commands of Part 1 that use relative paths (configs/, sessions/, tools/, apps/, pipeline/) run from real2sim/.
_astra_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export ASTRA_REPO="$_astra_root"
export PATH="$_astra_root/bin:$PATH"
export PYTHONPATH="$_astra_root/real2sim/src:$_astra_root/policy_development/src:$_astra_root/end_to_end/src${PYTHONPATH:+:$PYTHONPATH}"
unset _astra_root
