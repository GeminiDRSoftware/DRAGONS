

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
**The stable version is v3.2.2.**  


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13821517.svg)](https://doi.org/10.5281/zenodo.13821517) DRAGONS is distributed as a conda package, *dragons*, and it is 
included in the conda *gemini* stack.

Version 3.2 is recommend for the reduction of **imaging** data from Gemini's
current facility instruments: GMOS, NIRI, Flamingos-2, and GSAOI, for the
reduction of GMOS longslit spectroscopy data, and the reduction of GHOST data.

To reduce other types of Gemini spectroscopy data, please continue to use
the [Gemini IRAF package](https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software).

To install DRAGONS:

```
$ conda create -n dragons python=3.10 dragons ds9 "numpy<2"
```

You might need to add two relevant conda channels if you haven't already:

```
$ conda config --add channels conda-forge
$ conda config --add channels http://astroconda.gemini.edu/public
```


A list of changes since 3.1 can be found in the [Change Logs](https://dragons.readthedocs.io/en/v3.2.0/changes.html).

---
# What is DRAGONS
DRAGONS is a platform for the reduction and processing of astronomical data.
The DRAGONS meta-package includes an infrastructure for automation of the
processes and algorithms for processing of astronomical data, with focus on the
reduction of Gemini data.


---

# Documentation
Documentation on DRAGONS v3.2 is available on "readthedocs" at:

* https://dragons.readthedocs.io/en/v3.2.2/

There your will find manuals for Astrodata and the Recipe System, and hands-on
tutorials on reducing Gemini imaging data with DRAGONS.

Gemini users with imaging data to reduce should pick the tutorial discussing
the reduction of data from the appropriate instrument.

Software developers should start with the Astrodata and Recipe System
manuals.

---

# Setting up a development environment

To run checkouts, first set up a development conda environment.  This is what
we are using at this time for the `master` branch and the `release/3.2.x`
branches.

```
$ conda create -n dgdev3.10_20240401 python=3.10 astropy=6 astroquery matplotlib numpy psutil pytest python-dateutil requests scikit-image scipy sextractor sqlalchemy ds9 gwcs specutils sphinx sphinx_rtd_theme bokeh holoviews cython future astroscrappy=1.1 fitsverify imexam
$ conda activate dgdev3.10_20240401
$ pip install git+https://github.com/GeminiDRSoftware/GeminiObsDB.git@release/1.0.x
$ pip install git+https://github.com/GeminiDRSoftware/GeminiCalMgr.git@release/1.1.x
```
Dependencies change all the time and can break the development environment
or cause problems when conda tries to find a solution for the dependencies.
This not guaranteed to work flawlessly, you might have to adjust version
requirements.
