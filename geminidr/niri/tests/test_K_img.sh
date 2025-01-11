name=NIRI/K/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GN-2017B-FT-22-61:
"$script_dir/reduce_img.sh" "$name" N20180105 N20180107 || nerr=${nerr}1

end_test_set "$name" $nerr
