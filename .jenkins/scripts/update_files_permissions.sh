#!/usr/bin/env bash
# Updates the files permissions for refs/ and outputs/

chmod -R 775 reports/ || echo 1

echo "Update permissions inside ${DRAGONS_TEST_INPUTS} ---"
cd $DRAGONS_TEST_INPUTS; chmod -Rv 775 ./* || echo 1

echo "Update permissions inside ${DRAGONS_TEST_OUTPUTS} ---"
cd $DRAGONS_TEST_OUTPUTS; chmod -Rv 775 ./* || echo 1

echo "Update permissions inside ${DRAGONS_TEST_REFS} ---"
cd $DRAGONS_TEST_REFS; chmod -Rv 775 ./* || echo 1