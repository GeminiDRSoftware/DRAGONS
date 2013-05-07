
NICI_FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_FLAT.py

.. code-block:: python
    :linenos:

    class NICI_FLAT(DataClassification):
        name="NICI_FLAT"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all NICI flats 
            '''
        parent = "NICI_CAL"
        requirement = ISCLASS('NICI') & PHU(OBSTYPE='FLAT')
    
    newtypes.append( NICI_FLAT())



