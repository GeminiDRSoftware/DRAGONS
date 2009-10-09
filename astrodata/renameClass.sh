#!/bin/bash
# this find based perl script will replace all occurrances of a string with
# another string.  It could be modified to take arguments, but I put it here
# to remember it, it works well enough as is to do a class name change (would
# require a better re to change variable names 


find . -type f | xargs perl -pi -e 's/ReductionContext/ReductionContext/g'
