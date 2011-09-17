
NICI_DARK_CURRENT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_DARK_CURRENT

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_DARK_CURRENT.py

.. code-block:: python
    :linenos:

    
    class NICI_DARK_CURRENT(DataClassification):
        name="NICI_DARK_CURRENT"
        usage = "Applies to current dark current calibrations for the NICI instrument."
        parent = "NICI_DARK"
        requirement = ISCLASS('NICI') & PHU(OBSTYPE='DARK')
    
    newtypes.append(NICI_DARK_CURRENT())



