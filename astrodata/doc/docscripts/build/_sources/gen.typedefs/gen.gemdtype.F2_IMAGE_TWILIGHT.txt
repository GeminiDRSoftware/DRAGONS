
F2_IMAGE_TWILIGHT Classification Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :numbered:
    :maxdepth: 0
     
Classification
    F2_IMAGE_TWILIGHT

Source Location 
    ADCONFIG_Gemini/classifications/types/F2/gemdtype.F2_IMAGE_TWILIGHT.py

.. code-block:: python
    :linenos:

    class F2_IMAGE_TWILIGHT(DataClassification):
        name="F2_IMAGE_TWILIGHT"
        usage = """
            Applies to all imaging twilight flat datasets from the FLAMINGOS-2
            instrument
            """
        parent = "F2_IMAGE_FLAT"
        requirement = AND([  ISCLASS("F2_IMAGE_FLAT"),
                             OR([  PHU(OBJECT="Twilight"),
                                   PHU(OBJECT="twilight")  ])  ])
    
    newtypes.append(F2_IMAGE_TWILIGHT())



