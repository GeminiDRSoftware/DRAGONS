
FLAT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    FLAT

Source Location 
    ADCONFIG_Gemini/classifications/types/generic/gemdtype.FLAT.py

.. code-block:: python
    :linenos:

    class FLAT(DataClassification):
        name="FLAT"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all Gemini FLATS.
            '''
        parent = "CAL"
        requirement = OR(   ISCLASS("GMOS_FLAT"),
                            ISCLASS("NICI_FLAT"))
        
    newtypes.append( FLAT())



