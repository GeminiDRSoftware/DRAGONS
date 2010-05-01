#!/bin/bash

cd source/images_types
pwd
rm *.svg *.dot *.png

echo "without assignments shown..."
typelib GENERIC GEMINI RAW GMOS NIRI NICI GNIRS MICHELLE TRECS NIFS
echo "with assignments shown..."
typelib -a GENERIC GEMINI RAW GMOS NIRI NICI GNIRS MICHELLE TRECS NIFS

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

