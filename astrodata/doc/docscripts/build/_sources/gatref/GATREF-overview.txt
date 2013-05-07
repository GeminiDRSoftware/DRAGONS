Overview
--------

In order to configure astrodata with the Gemini data types and descriptors, the
astrodata package must be able to find the ADCONFIG package in which they are
defined. The ADCONFIG packages must be within a subdirectory called
"ADCONFIG_<whatever>", the Gemini package is "ADCONFIG_Gemini". The astrodata
package will search the PYTHONPATH for these packages.  While they do contain
python code, they are not meant to be directly imported, and PYTHONPATH is used
to make installation simpler. One can also set ADCONFIGPATH.  Note: the PATH
environment variable point to the parent directory, which contains the
"ADCONFIG_..." subdirectory.
