

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
**The stable version is v4.0.0.**  


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15226659.svg)](https://doi.org/10.5281/zenodo.15226659) DRAGONS is distributed as a conda package, *dragons*, and it is 
included in the conda *gemini* stack.

Version 4.0 is recommend for the reduction of **imaging** data from Gemini's
current facility instruments: GMOS, NIRI, Flamingos-2, and GSAOI, for the
reduction of GMOS and GNIRS **longslit spectroscopy** data, and the reduction 
of GHOST data.

To reduce other types of Gemini spectroscopy data, please continue to use 
the [Gemini IRAF package](https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software).

To install DRAGONS:

```
$ conda create -n dragons python=3.12 dragons ds9
```

You might need to add two relevant conda channels if you haven't already:

```
$ conda config --add channels conda-forge
$ conda config --add channels http://astroconda.gemini.edu/public
```


A list of changes since 3.2 can be found in the [Change Logs](https://dragons.readthedocs.io/en/v4.0.0/changes.html).

---
# What is DRAGONS
DRAGONS is a platform for the reduction and processing of astronomical data.
The DRAGONS meta-package includes an infrastructure for automation of the
processes and algorithms for processing of astronomical data, with focus on the 
reduction of Gemini data.


---

# Documentation
Documentation on DRAGONS v4.0 is available on "readthedocs" at:

* https://dragons.readthedocs.io/en/v4.0.0/

There your will find manuals for Astrodata and the Recipe System, and hands-on
tutorials on reducing Gemini imaging data with DRAGONS.

Gemini users with data should pick the tutorial discussing
the reduction of data from the appropriate instrument and mode.  

Software developers should start with the Astrodata and Recipe System
manuals.

---

# Setting up a development environment

To run checkouts, first set up a development conda environment.  This is what
we are using at this time for the `master` branch and the `release/4.0.x` 
branches.

```
$ conda create -n dgdev3.12_20250520 python=3.12 "astropy>=6" astroquery matplotlib "numpy<2" psutil pytest python-dateutil requests scikit-image scipy sextractor "sqlalchemy>=2.0.0" "gwcs>=0.15,<=0.22.1" specutils "bokeh>=3" holoviews cython future "astroscrappy>=1.1" fitsverify jsonschema ds9 jupyter ipympl imexam sphinx sphinx_rtd_theme objgraph
$ conda activate dgdev3.12_20250520
$ pip install git+https://github.com/GeminiDRSoftware/FitsStorage.git@v3.4.x
```
Dependencies change all the time and can break the development environment
or cause problems when conda tries to find a solution for the dependencies. 
This not guaranteed to work flawlessly, you might have to adjust version
requirements.

