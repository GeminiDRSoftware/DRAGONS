.. customize:

*******************************
Customizing the Data Processing
*******************************

Getting Information about Recipes and Primitives
================================================
get list of recipes, get list of primitives
where are the recipes stored, where are the primitives stored

Adding or Modifying a Recipes
=============================

add/remove primitives, adjust input parameters.
discussion of addToList and getList might be appropriate here.

Writing and Using your own Primitive
====================================

use template

Focus on building on top of the astrodata_Gemini package.
Mention that if new instrument, ADCONFIG needs to be updated (see astrodata manual)
Also, if new instrument, the primitive will have to belong in a new class, won't it.

question: can a primitive be stored outside of a RECIPE_* directory? and call with -r?
question: where does one put the primitive?  which class, which module?
question: how does inheritance works.

next section has to be about creating a new RECIPE_* in astrodata_* package.

mini, really mini discussion about ReductionContext as this is how input files
are passed into the primitive.
