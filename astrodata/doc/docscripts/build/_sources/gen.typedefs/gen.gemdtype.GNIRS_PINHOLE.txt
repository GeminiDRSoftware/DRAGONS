
GNIRS_PINHOLE Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GNIRS_PINHOLE

Source Location 
    ADCONFIG_Gemini/classifications/types/GNIRS/gemdtype.GNIRS_PINHOLE.py

.. code-block:: python
    :linenos:

    
    class GNIRS_PINHOLE(DataClassification):
        name="GNIRS_PINHOLE"
        usage = "Applies to GNIRS Pinhole mask calibration observations"
        parent = "GNIRS"
        requirement = AND(ISCLASS('GNIRS'), PHU(OBSTYPE='FLAT'), OR( PHU(SLIT='LgPinholes_G5530'), PHU(SLIT='SmPinholes_G5530') ))
    
    newtypes.append(GNIRS_PINHOLE())



