#!/usr/bin/env bash
# Quick-and-dirty test for imaging reduction (command-line interface) until we
# re-establish a proper test framework for DRAGONS. This is mostly unrelated to
# existing Python tests. It expects raw data in $GEMPYTHON_TESTDATA/$test_dir
# (or failing that the CWD) and comparison files in "ref/" under the same dir.
# Some of this shell stuff should be tidied up with Python utilities.

# Get args. Filename patterns for long & short darks are user-specified for
# lack of a better way to distinguish them for the time being (unless we assume
# there are 10 of each, which is bound to break occasionally):
if [ $# -lt 4 -o $# -gt 5 ]; then
    echo Usage: $(basename "$0") \
         "test_dir bpm_dark_pattern targ_name std_name [maxim]" >& 2
    echo "  (where pattern is a filename regex such as N20180107)" >& 2
    exit 1
fi
test_dir=$1         # (sh_functions expects this variable name)
bpm_dark_patt=$2
targ_name=$3
std_name=$4
maxim=$5            # process first N frames, due to stacking memory limit
[ -z "$maxim" ] && maxim=999999

# Load common sh functions for testing CLI:
geminidr_dir=$(cd $(dirname "$0")/../..; pwd)
. "$geminidr_dir/gemini/test/sh_functions"

# Generate file lists for calibrations:
dataselect "$data_dir/*.fits" --tags DARK,RAW > "$work_dir/darks.lis"
grep -E "$bpm_dark_patt" darks.lis | sort | head -5 > darks_bpm.lis

dataselect "$data_dir/*.fits" --tags FLAT,RAW,LAMPON > "$work_dir/flatson.lis"
dataselect "$data_dir/*.fits" --tags FLAT,RAW,LAMPOFF > "$work_dir/flatsoff.lis"
grep -Ev "^[ \t]*#" flatson.lis | sort | head -5 > flats.lis
grep -Ev "^[ \t]*#" flatsoff.lis | sort | head -5 >> flats.lis

# List std data:
dataselect "$data_dir/*.fits" --tags IMAGE,RAW --xtags FLAT --expr="object=='$std_name'" > "$work_dir/std.lis"

# List science data:
dataselect "$data_dir/*.fits" --tags IMAGE,RAW --xtags FLAT --expr="object=='$targ_name'" > "$work_dir/sci.lis"
grep -Ev "^[ \t]*#" sci.lis | sort | head -$maxim > sciset.lis

# This is not actually needed since makeProcessedBPM reduces the only dark?
# Reduce the darks & put them in the calibration-matching database:
for darklist in darks_*.lis; do
    reduce "@$darklist"
    # caldb add "$(first_filename $darklist dark)"
    caldb add "$(last_result_filename dark)"
done

# Generate BPM:
reduce @flats.lis @darks_bpm.lis -r makeProcessedBPM

# Save BPM name from the log, to allow repetition with bits commented out
# (this must immediately follow the above line):
last_result_filename bpm > bpmname

bpm=$(cat bpmname)

# Reduce flats. The order of the files in the list doesn't get preserved, so
# have to figure out the output name from the log.
reduce @flats.lis -p addDQ:user_bpm=${bpm}
caldb add "$(last_result_filename flat)"

# Reduce std:
reduce @std.lis -p addDQ:user_bpm=${bpm}

# Reduce science data:
reduce @sciset.lis -p addDQ:user_bpm=${bpm}

# Check the final result & return status. Since there's no stack here, just
# spot check the last file:
compare_file $(last_result_filename skySubtracted)
