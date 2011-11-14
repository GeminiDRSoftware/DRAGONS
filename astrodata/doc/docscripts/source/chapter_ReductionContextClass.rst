ReductionContext Class Reference
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The following is information about the ReductionContext class. When writing
primitives the reduction context is the sole argument (generally named ``rc`` by
Gemini conventions).  This object is used by the primitive to both get inputs
and store outputs, as well as communication with certain privileged subsystems
like calibration queries.

Parameter and Dictionary Features
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

The "in" operator: contains(..)
#################################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.__contains__

Dataset Streams: Input and Output Datasets
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

get_inputs(..)
###############

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_inputs

get_inputs_as_astrodata(..)
###########################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_inputs_as_astrodata

get_inputs_as_filenames(..)
############################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_inputs_as_filenames

get_stream(..)
###############

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_stream

get_reference_image(..)
########################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_reference_image

report_output(..)
########################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.report_output

switch_stream(..)
########################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.switch_stream

ADCC Services
@@@@@@@@@@@@@@

Calibrations
@@@@@@@@@@@@@@

add_cal(..)
############

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.add_cal

get_cal(..)
############

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_cal

rq_cal(..)
############

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.rq_cal

Stacking
@@@@@@@@@@

rq_stack_get(..)
#################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.rq_stack_get

rq_stack_update(..)
####################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.rq_stack_update

Lists
@@@@@@

append_list(..)
###################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.append_list

get_list(..)
###################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.get_list

rq_stack_update(..)
###################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.rq_stack_update

Utility
@@@@@@@@

prepend_names(..)
###################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.prepend_names


run(..)
###################

.. toctree::

.. automethod:: astrodata.RecipeManager.ReductionContext.run


