name=F2/H

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GS-2017B-Q-41-43:
"$script_dir/reduce_img_nodarks.sh" F2/H/raw "S201801..S00[0-1]" 48 || nerr=${nerr}1

end_test_set "$name" $nerr
