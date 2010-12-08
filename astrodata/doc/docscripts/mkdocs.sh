#!/bin/bash

./grabWiki.py
./sphbuild.sh

acroread build/_latex_build/astrodatadocumentation.pdf
