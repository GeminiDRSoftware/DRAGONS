
NICI_ASDI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_ASDI

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_ASDI.py

.. code-block:: python
    :linenos:

    
    class NICI_ASDI(DataClassification):
        name="NICI_ASDI"
        usage = "Applies to imaging datasets from the NICI instrument."
        parent = "NICI"
        # DICHROIC PHU keyword value contains the string '50/50'
        requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?50/50.*?" }) & \
                      PHU(CRMODE='FIXED') & PHU({'{prohibit}OBSTYPE': 'FLAT'})
    
    newtypes.append(NICI_ASDI())



