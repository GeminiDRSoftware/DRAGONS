#!/usr/bin/env bash
name=GSAOI/H/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GS-2018A-Q-119-19:
"$script_dir/reduce_img.sh" "$name" "S20160219S01" AT2018ec 9146 20 || nerr=${nerr}1

end_test_set "$name" $nerr
