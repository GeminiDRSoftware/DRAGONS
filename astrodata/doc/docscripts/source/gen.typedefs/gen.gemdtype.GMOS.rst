
GMOS Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    GMOS

Source Location 
    ADCONFIG_Gemini/classifications/types/GMOS/gemdtype.GMOS.py

.. code-block:: python
    :linenos:

    class GMOS(DataClassification):
        name="GMOS"
        # this a description of the intent of the classification
        # to what does the classification apply?
        usage = '''
            Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
            '''
            
        parent = "GEMINI"
        requirement = ISCLASS("GMOS_N") | ISCLASS("GMOS_S")
        # equivalent to...
        #requirement = OR(   
        #                    ClassReq("GMOS_N"), 
        #                    ClassReq("GMOS_S")
        #                    )
    
    newtypes.append( GMOS())



