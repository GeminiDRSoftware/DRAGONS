
NICI_CAL Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    NICI_CAL

Source Location 
    ADCONFIG_Gemini/classifications/types/NICI/gemdtype.NICI_CAL.py

.. code-block:: python
    :linenos:

    class NICI_CAL(DataClassification):
        name="NICI_CAL"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all NICI flats 
            '''
        parent = "NICI"
        requirement = ISCLASS('NICI_FLAT') | ISCLASS('NICI_DARK')
    
    newtypes.append( NICI_CAL())



