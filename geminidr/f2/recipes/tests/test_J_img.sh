name=F2/J_test3/raw

# Source a couple of helper functions etc.:
script_dir=$(cd $(dirname "$0"); pwd)
. "$script_dir/../../gemini/test/sh_functions"

start_test_set "$name"

# GS-F2-RECOM13-RUN-3-331; same data as RTF test 3:
"$script_dir/reduce_img.sh" "$name" "S20130719S052" "S20130719S055" || nerr=${nerr}1

end_test_set "$name" $nerr
