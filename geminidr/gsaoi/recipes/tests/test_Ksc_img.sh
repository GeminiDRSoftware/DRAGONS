#!/usr/bin/env bash
name=GSAOI/Ksc/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GS-2018A-Q-130-23:
# The data have been culled by hand (moved to extra/ dir) on James's disk, due to the awkward dither pattern:
"$script_dir/reduce_img_every_n.sh" "$name" "Sgr A* - off-centered" 9182 1 || nerr=${nerr}1

end_test_set "$name" $nerr
