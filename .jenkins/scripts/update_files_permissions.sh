#!/usr/bin/env bash
# Updates the files permissions for refs/ and outputs/

chmod -R 775 reports/ || echo 1

chmod -Rv 775 $DRAGONS_TEST_INPUTS || echo 1

chmod -Rv 775 $DRAGONS_TEST_OUTPUTS || echo 1

chmod -Rv 775 $DRAGONS_TEST_REFS || echo 1
