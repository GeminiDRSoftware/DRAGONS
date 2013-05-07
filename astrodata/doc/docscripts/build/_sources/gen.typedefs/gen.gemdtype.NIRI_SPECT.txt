
NIRI_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIRI_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/NIRI/gemdtype.NIRI_SPECT.py

.. code-block:: python
    :linenos:

    
    class NIRI_SPECT(DataClassification):
        name="NIRI_SPECT"
        usage = "Applies to any spectra from the NIRI instrument."
        parent = "NIRI"
        requirement = ISCLASS('NIRI') & PHU(FILTER3='(.*?)grism(.*?)')
    
    newtypes.append(NIRI_SPECT())



