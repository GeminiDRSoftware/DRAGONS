

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
**The stable version is v3.0.0.**  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4025470.svg)](https://doi.org/10.5281/zenodo.4025470) DRAGONS is distributed as a conda package, *dragons*, and it is 
included in the conda *gemini* stack.

Version 3.0.0 is recommend of the reduction of **imaging** data from Gemini's
current facility instruments: GMOS, NIRI, Flamingos-2, and GSAOI.

There is no science quality spectroscopy support in this release.  To reduce 
Gemini spectroscopy data, please continue to use the 
[Gemini IRAF package](https://www.gemini.edu/sciops/data-and-results/processing-software).

There is however support for **quicklook** reduction of GMOS longslit spectroscopic
data.  The products have not been verified for scientific accuracy.

To install:

```
$ conda create -n dragons python=3.7 dragons stsci
```

You might need to add two relevant conda channels if you haven't already:

```
$ conda config --add channels http://ssb.stsci.edu/astroconda
$ conda config --add channels http://astroconda.gemini.edu/public
```


A list of changes since 2.1.1. can be found in the [Change Logs](https://dragons.readthedocs.io/en/v3.0.0/changes.html).

---
# What is DRAGONS
DRAGONS is a platform for the reduction and processing of astronomical data.
The DRAGONS meta-package includes an infrastructure for automation of the
processes and algorithms for processing of astronomical data, with focus on the 
reduction of Gemini data.


---

# Documentation
Documentation on DRAGONS v3.0 is available on "readthedocs" at:

* https://dragons.readthedocs.io/en/stable

There your will find manuals for Astrodata and the Recipe System, and hands-on
tutorials on reducing Gemini imaging data with DRAGONS.

Gemini users with imaging data to reduce should pick the tutorial discussing
the reduction of data from the appropriate instrument.  

Software developers should start with the Astrodata and Recipe System
manuals.

