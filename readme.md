

<table border="0">
<tr>
  <td rowspan="2"><img src="./graphics/DRAGONS-Iconblue.png" width="100" height="100"></td>
  <td><font size="18">DRAGONS</font></td>
</tr>
<tr>
  <td>Data Reduction for Astronomy <br>from Gemini Observatory North and South</font></td>
</tr>
</table>

# Current Status
**The stable version is v2.1.0.**  This is the first publicly released version
of DRAGONS.  It is distributed as a conda package, *dragons*, and it is 
included in the conda *gemini* stack.

Version 2.1.0 is recommend of the reduction of **imaging** data from Gemini's
current facility instruments: GMOS, NIRI, Flamingos-2, and GSAOI.

There is no spectroscopy support in this release.  To reduce Gemini spectroscopy
data, please continue to use the [Gemini IRAF package](https://www.gemini.edu/sciops/data-and-results/processing-software).

To install:

$ conda create -n geminiconda python=3.6 gemini stsci


---
# What is DRAGONS
DRAGONS is a platform for the reduction and processing of astronomical data.
The DRAGONS meta-package includes an infrastructure for automation of the
processes and algorithms for processing of astronomical data, with focus on the 
reduction of Gemini data.


---

# Documentation
Documentation on DRAGONS v2.1 is available on "readthedocs" at:

* https://dragons.readthedocs.io/en/stable

There your will find manuals for Astrodata and the Recipe System, and hands-on
tutorials on reducing Gemini imaging data with DRAGONS.

Gemini users with imaging data to reduce should pick the tutorial discussing
the reduction of data from the appropriate instrument.  

Software developers should start with the Astrodata and Recipe System
manuals.