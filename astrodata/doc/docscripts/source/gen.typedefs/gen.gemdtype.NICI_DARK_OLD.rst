
NICI_DARK_OLD Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_DARK_OLD

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_DARK_OLD.py

.. code-block:: python
    :linenos:

    
    class NICI_DARK_OLD(DataClassification):
        name="NICI_DARK_OLD"
        usage = "Applies to OLD NICI dark current calibration datasets."
        parent = "NICI_DARK"
        requirement = ISCLASS('NICI') & PHU(OBSTYPE='FLAT',
                                            GCALSHUT='CLOSED')
    
    newtypes.append(NICI_DARK_OLD())



