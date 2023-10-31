

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
**The stable version is v3.1.0.**  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7776065.svg)](https://doi.org/10.5281/zenodo.7776065) 
DRAGONS is distributed as a conda package, *dragons*, and it is 
included in the conda *gemini* stack.

Version 3.1 is recommend for the reduction of **imaging** data from Gemini's
current facility instruments: GMOS, NIRI, Flamingos-2, and GSAOI, and for the
reduction of GMOS longslit spectroscopy data.

To reduce other types of Gemini spectroscopy data, please continue to use 
the [Gemini IRAF package](https://gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software).

To install DRAGONS:

```
`$ conda create -n dragons python=3.10 dragons ds9
```

You might need to add two relevant conda channels if you haven't already:

```
$ conda config --add channels conda-forge
$ conda config --add channels http://astroconda.gemini.edu/public
```


A list of changes since 3.0 can be found in the [Change Logs](https://dragons.readthedocs.io/en/v3.1.0/changes.html).

---
# What is DRAGONS
DRAGONS is a platform for the reduction and processing of astronomical data.
The DRAGONS meta-package includes an infrastructure for automation of the
processes and algorithms for processing of astronomical data, with focus on the 
reduction of Gemini data.


---

# Documentation
Documentation on DRAGONS v3.1 is available on "readthedocs" at:

* https://dragons.readthedocs.io/en/v3.1.0/

There your will find manuals for Astrodata and the Recipe System, and hands-on
tutorials on reducing Gemini imaging data with DRAGONS.

Gemini users with imaging data to reduce should pick the tutorial discussing
the reduction of data from the appropriate instrument.  

Software developers should start with the Astrodata and Recipe System
manuals.

---

# Citing DRAGONS

If you are using DRAGONS for your project, we ask that you please cite the 
following paper:

* [K. Labrie et al 2023 Res. Notes AAS 7 214](https://iopscience.iop.org/article/10.3847/2515-5172/ad0044) ([BibTex](https://iopscience.iop.org/export?type=article&doi=10.3847/2515-5172/ad0044&exportFormat=iopexport_bib&exportType=abs&navsubmit=Export+abstract))

The DOI for DRAGONS version 3.1.0 is:

* [10.5281/zenodo.7776065](https://zenodo.org/record/7776065) ([BibTex](https://zenodo.org/record/7776065/export/hx))

If you are using AASTeX and plan to submit an article to one of the AAS journals,
we recommend adding a `\software` tag to your manuscript that cites DRAGONS and
the specific version you have used. For example:

```
\software{DRAGONS \citep{dragonsRNAAS_2023}, \cite[Version 3.1.0]{dragons3.1.0}
```