
GNIRS_SPECT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GNIRS_SPECT

Source Location 
    ADCONFIG_Gemini/classifications/types/GNIRS/gemdtype.GNIRS_SPECT.py

.. code-block:: python
    :linenos:

    
    class GNIRS_SPECT(DataClassification):
        name="GNIRS_SPECT"
        usage = "Applies to any SPECT dataset from the GNIRS instrument."
        parent = "GNIRS"
        requirement = ISCLASS('GNIRS') & PHU(ACQMIR='Out')
    
    newtypes.append(GNIRS_SPECT())



