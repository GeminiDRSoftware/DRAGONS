Gemini Type Graphs
-------------------

.. toctree::
    :numbered:

Gemini dataset types, or "AstroData Types" include three distinct trees.

+ The GEMINI tree which relates to instruments and instrument-modes.
+ The RAW tree which is single descendant, and represents processing status.
  These types are presented mixed with the "typological" types through 
  the getTypes interface, but can also be retrieves through the getStatus call.
+ A generic type tree which relates to the GEMINI tree (i.e. IMAGE is any 
  of the specific INSTR_IMAGE types.

The graphs below are derived from the ADCONFIG_Gemini AstroData Configuration
Package as of |today|, and show descriptor and primitive set assignments when present.


GENERIC
~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1


The GENERIC type tree relates to abstract data modes which may be used to apply recipes generic to these modes.  If generic primitives
can be written, these too can be shared, but it is also possible to assign the primitives at instrument-mode specific granularity as
well. That is, one can provide generic code at the levels where it makes sense, and type-specific code when that makes more sense or is
expedient.  We generally visualize an incremental development process where new instrument-modes are first supported  in type-specific code where
changes to the system are isolated and don't affect other processing. Subsequently this code can be integrated into, or merely replaced
by, more general algorithms that use other AstroData features to abstract away incidental differences between datasets from different
devices.

.. figure:: images_types/GENERIC-tree-pd.png
    :width: 90%
    
    The Gemini GENERIC Type Tree

GEMINI
~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

The complete tree of instrument-mode related typological classifications all
descend from the GEMINI type, which means of course, the data was from a
GEMINI telescope. The figure is difficult to read as all types are present, and
will get more so as the instrument trees are filled out.  The instrument related
graphs are more informative, but this gives and idea of the overall taxology.


.. figure:: images_types/GEMINI-tree-pd.png
    :width: 100%

    The Gemini GEMINI Type Tree


GMOS
~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

GMOS is an optical instrument with an imaging mode, an IFU, and a multi-object
spectrograph. We have a complete first revision of the GMOS tree.

.. figure:: images_types/GMOS-tree-pd.png
    :width: 90%

    The Gemini GMOS Type Tree

GNIRS
~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

GNIRS is a near-infrared spectroscopy instrument currently under repair. The
tree is just a stub, which recognizes data from GNIRS.

.. figure:: images_types/GNIRS-tree-pd.png
    :width: 20%

    The Gemini GNIRS Type Tree

NICI
~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

NICI is the Near-Infrared Coronagraphic Imager, used at Gemini South. We have a
preliminary (development) first revision of the NICI type tree.

.. figure:: images_types/NICI-tree-pd.png
    :width: 75%

    The Gemini NICI Type Tree

NIFS
~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1


NIFS is a Near-Infrared Integral Field Spectrometer  uses at Gemini North.  We
have a minimal tree in place for NIFS, which recognizes IMAGE and SPECT types.

.. figure:: images_types/NIFS-tree-pd.png
    :width: 30%

    The Gemini NIFS Type Tree

NIRI
~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

NIRI is a Near-Infrared Imager in use at Gemini North. We have a minimal tree in
place for NIRI, which recognizes IMAGE and SPECT types.

.. figure:: images_types/NIRI-tree-pd.png
    :width: 30%

    The Gemini NIRI Type Tree

MICHELLE
~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

MICHELLE is a mid-infrared image and sepctrometer. We have a minimal tree in
place for MICHELLE, which recognizes IMAGE and SPECT types.

.. figure:: images_types/MICHELLE-tree-pd.png
    :width: 30%

    The Gemini MICHELLE Type Tree

RAW
~~~~~

.. toctree::
    :numbered:
    :maxdepth: 1

The RAW tree contains inherent sequencing, what are show as children are new
forms of the data... that is, some transformation(s) will make the data
recognized only as the child.  These types can be used to check the state of
processing, and can have entirely generic recipes associated, as may be needed
by some pipelines.

.. figure:: images_types/RAW-tree-pd.png
    :width: 60%

    The Gemini RAW Type Tree

TRECS
~~~~~~

.. toctree::
    :numbered:

TRECS is a Thermal-Region Camera Spectograph, a mid-infrared imager and
long-slit spectrograph built  for Gemini South. We have a minimal tree in place
for TRECS, which recognizes IMAGE and SPECT.

.. figure:: images_types/TRECS-tree-pd.png
    :width: 30%

    The Gemini TRECS Type Tree
