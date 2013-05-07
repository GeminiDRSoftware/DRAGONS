
NICI_SDI Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_SDI

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_SDI.py

.. code-block:: python
    :linenos:

    
    class NICI_SDI(DataClassification):
        name="NICI_SDI"
        usage = "Applies to imaging datasets from the NICI instrument."
        parent = "NICI"
        # DICHROIC PHU keyword value contains the string '50/50'
        requirement = ISCLASS('NICI') & PHU( {'{re}.*?DICHROIC': ".*?50/50.*?" }) & \
                      PHU(CRMODE='FOLLOW') & PHU({'{prohibit}OBSTYPE':'FLAT'})
    
    newtypes.append(NICI_SDI())



