#!/bin/bash

cd source/images_types
pwd
rm *.svg *.dot *.png

echo "without assignments shown..."
typelib GEMINI GENERIC GMOS GNIRS NICI NIFS NIRI MICHELLE RAW TRECS 
echo "with assignments shown..."
typelib -a GEMINI GENERIC GMOS GNIRS NICI NIFS NIRI MICHELLE RAW TRECS 

echo "special graphs"
typelib -a GMOS_IMAGE

# typelib GENERIC
# typelib GEMINI
# typelib RAW
# typelib GMOS
# typelib NIRI
# typelib NICI
# typelib GNIRS
# typelib MICHELLE
# typelib TRECS
# typelib NIFS

