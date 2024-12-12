#!/usr/bin/env bash
name=GSAOI/J/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GS-2018A-Q-230-13:
"$script_dir/reduce_img_not_bpm.sh" "$name" "WISE0833+0052" 9146 || nerr=${nerr}1

end_test_set "$name" $nerr
