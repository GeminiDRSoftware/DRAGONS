#!/usr/bin/env bash

if [ ! -d "$TEST_PATH" ]; then
    echo "Test path does not exists: $TEST_PATH"
    mkdir -p $TEST_PATH
fi

if [ ! -d "$TEST_PATH/GMOS/" ]; then
    echo "Test path does not exists: $TEST_PATH/GMOS/"
    mkdir -p $TEST_PATH
fi

file="test_files"
while IFS= read -r line
do
    printf '%s\n' "$line"
done < "$file"
