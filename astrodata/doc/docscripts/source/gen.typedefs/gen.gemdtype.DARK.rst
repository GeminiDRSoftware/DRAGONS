
DARK Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    DARK

Source Location 
    ADCONFIG_Gemini/classifications/types/generic/gemdtype.DARK.py

.. code-block:: python
    :linenos:

    class DARK(DataClassification):
        name="DARK"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to any dataset that is a Gemini dark current calibration.
        '''
        parent = "CAL"
        requirement = OR(ISCLASS("NICI_DARK"),
                        ISCLASS("GMOS_DARK"))
    
    newtypes.append( DARK())



