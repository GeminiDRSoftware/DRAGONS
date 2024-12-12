# Quick-and-dirty test for imaging reduction (command-line interface) until we
# re-establish a proper test framework for DRAGONS. This is mostly unrelated to
# existing Python tests. It expects raw data in $GEMPYTHON_TESTDATA/$test_dir
# (or failing that the CWD) and comparison files in "ref/" under the same dir.
# Some of this shell stuff should be tidied up with Python utilities.

# Get args. Filename patterns for long & short darks are user-specified for
# lack of a better way to distinguish them for the time being (unless we assume
# there are 10 of each, which is bound to break occasionally):
if [ $# -lt 3 -o $# -gt 4 ]; then
    echo Usage: $(basename "$0") \
         "test_dir short_dark_pattern long_dark_pattern [maxim]" >& 2
    echo "  (where pattern is a filename regex such as N20180107)" >& 2
    exit 1
fi
test_dir=$1         # (sh_functions expects this variable name)
short_dark_patt=$2
long_dark_patt=$3
maxim=$4            # process first N frames, due to stacking memory limit
[ -z "$maxim" ] && maxim=999999

# Load common sh functions for testing CLI:
geminidr_dir=$(cd $(dirname "$0")/../..; pwd)
. "$geminidr_dir/gemini/test/sh_functions"

# Generate file lists for calibrations:
(cd "$data_dir"; typewalk --tags DARK RAW -o "$work_dir/darks.lis")
grep -Ev "^[ \t]*#" darks.lis | sort > sorted_darks.lis
grep -E "$short_dark_patt" sorted_darks.lis > darks_short.lis
grep -E "$long_dark_patt" sorted_darks.lis > darks_sci.lis

(cd "$data_dir"; typewalk --tags FLAT RAW -o "$work_dir/flats.lis")

# List science data; would need splitting somehow if there were also a std:
(cd "$data_dir"; typewalk --tags IMAGE RAW --xtags FLAT -o "$work_dir/sci.lis")
grep -Ev "^[ \t]*#" sci.lis | sort | head -$maxim > sciset.lis

# Reduce the darks & put them in the calibration-matching database:
for darklist in darks_*.lis; do
    reduce "@$darklist"
    # caldb add "$(first_filename $darklist dark)"
    caldb add "$(last_result_filename dark)"
done

# Generate BPM:
reduce @flats.lis @darks_short.lis -r makeProcessedBPM

# Save BPM name from the log, to allow repetition with bits commented out
# (this must immediately follow the above line):
last_result_filename bpm > bpmname

bpm=$(cat bpmname)

# Reduce flats. The order of the files in the list doesn't get preserved, so
# have to figure out the output name from the log.
reduce @flats.lis -p addDQ:user_bpm=${bpm}
caldb add "$(last_result_filename flat)"

# Reduce science data:
reduce @sciset.lis -p alignAndStack:save=True -p addDQ:user_bpm=${bpm}

# Check the final result & return status:
compare_file $(last_result_filename stack)
