
OSCIR Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    OSCIR

Source Location 
    ADCONFIG_Gemini/classifications/types/OSCIR/gemdtype.OSCIR.py

.. code-block:: python
    :linenos:

    
    class OSCIR(DataClassification):
        name="OSCIR"
        usage = "Applies to datasets from the OSCIR instrument"
        parent = "GEMINI"
        requirement = OR(PHU(INSTRUME='oscir'), PHU(INSTRUME='OSCIR'))
    
    newtypes.append(OSCIR())



