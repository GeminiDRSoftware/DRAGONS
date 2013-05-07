
NIFS_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NIFS_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/NIFS/gemdtype.NIFS_SPECT.py

.. code-block:: python
    :linenos:

    
    class NIFS_SPECT(DataClassification):
        name="NIFS_SPECT"
        usage = "Applies to any spectroscopy dataset from the NIFS instrument."
        parent = "NIFS"
    
        requirement = ISCLASS("NIFS") & PHU( FLIP='Out')
    
    newtypes.append(NIFS_SPECT())



