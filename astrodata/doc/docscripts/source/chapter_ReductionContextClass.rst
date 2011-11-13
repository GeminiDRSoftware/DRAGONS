ReductionContext Class Reference
!!!!!!!!!!!!!!!!!!!!!!!!!

.. toctree::
    
The following is information about the ReductionContext class. When writing
primitives the reduction context is the sole argument (generally named ``rc`` by
Gemini conventions).  This object is used by the primitive to both get inputs
and store outputs, as well as communication with certain privileged subsystems
like calibration queries.

.. autoclass:: astrodata.data.ReductionContext

Basic Functions
@@@@@@@@@@@@@@@

Parameter and Dictionary Features
##################################

The "in" operator: contains(..)
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.__contains__

Dataset Streams: Input and Output Datasets
###########################################


ADCC Services
##############

Calibrations
$$$$$$$$$$$$$$$$

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.add_cal

Stacking
$$$$$$$$$$

Lists
$$$$$$
