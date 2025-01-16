name=NIRI/J/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GN-2018A-Q-130-37:
"$script_dir/reduce_img.sh" "$name" N20180113S N20171001S || nerr=${nerr}1

end_test_set "$name" $nerr
