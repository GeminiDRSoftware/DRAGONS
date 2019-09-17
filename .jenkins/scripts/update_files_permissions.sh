#!/usr/bin/env bash
# Updates the files permissions for refs/ and outputs/

cd $DRAGONS_TEST_OUTPUTS; chmod -Rv 775 ./* || echo 1
cd $DRAGONS_TEST_REFS; chmod -Rv 775 ./* || echo 1