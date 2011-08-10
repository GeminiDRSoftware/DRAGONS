


AstroData Lexicon
-----------------


astrodata Grammar
~~~~~~~~~~~~~~~~~


+ AstroDataType >The term refers to a specific type of data, such as
that associated with a particular instrument mode (e.g. GMOS_IMAGE) or
reduced type (IFU_CUBED). The associated action is execution of a
criteria for correct classification as the given type. < li>
+ Descriptors >The term refers to high level metadata which conforms
across datatypes, as opposed to the lower level metadata in file and
extension headers, from which the higher level "descriptors"
will be calculated. The associated action is the algorithmic
production or calculation of the high level metadata. < li>
+ Structures >The term refers to a certain data layout, such as
hierarchical nesting, and member names to associate these file
elements with, so that, for example, a particular extension type in
the source MEF will be given a particular member name driven by the
structure definition. E.g. the extension named (MDF) can be assigned
to the ad.mdf member for type SPECT. < li>
+ Primitives >The term refers to dataset transformations, and should
  be a scientifically meaningful term. This latter requirement ensures
  that recipes will be scientifically meaningful, since they are
  composed of series of primitives. Ideally, recipes then contain just
  the scientifically meaningful structure. More technically, and
  broadly, the primitive represents a transformation between different
  Reduction Context states, requiring some bounding input state and
  producing an output state. The action associated is, of course, to
  perform the transformation. < li>



AstroTypes, Primitive Sets and Descriptor Calculators
-----------------------------------------------------

<html><img alt="GMOS AstroData Type Tree"
style="margin:.5em;padding:.5em; border:1px black solid" width = "90%"
src="`http://ophiuchus.hi.gemini.edu/ADMANUALSOURCE/images_types/GMOS-
tree.png <http://ophiuchus.hi.gemini.edu/ADMANUALSOURCE/images_types
/GMOS-tree.png>`__"/></html>

The core term, upon which everything else relies are the AstroData
types. To the right is a directed graph of the GMOS type tree. The
"house" shaped elements are the types, with solid lines pointing to
child types.

Blue and red rectangular nodes point dashed lines toward the types
where primitive sets and descriptor calculators are assigned.

