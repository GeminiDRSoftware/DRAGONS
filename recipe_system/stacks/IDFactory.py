#
#                                                                   IDFactory.py
#------------------------------------------------------------------------------ 
def make_safe_id(spaced_id):
    return spaced_id.replace(" ", "_")

def generate_stackable_id(dataset):
    """
    Generate an ID from which all similar stackable data will have in common.
    
    @param dataset: AstroData object
    @type dataset: <AstroData> instance
    
    @return: A stackable id.
    @rtype:  <str>

    """
    _v = "1_0"
    ID = "{}{}".format(_v, dataset.group_id())
    return make_safe_id(ID)

def generate_astro_data_id(dataset):
    """
    An id to be used to identify AstroData types. This is used for:

    1) Calibrations:
    
    Let's say a recipe performs
    
    getProcessedBias

    prepare
    biasCorrect
    
    Because of the prepare step, the calibration key determined at 
    getProcessedBias will not match biasCorrect because (N2009..., bias) will 
    not match (gN2009..., bias). By using an astro_id, you can avoid this issue 
    as you will have (DATALAB, bias). So, any steps inbetween getProcessedBias
    and biasCorrect will have no impact.
    
    2) Fringe:
    
    Fringe uses this as a FringeID, which is based off the first input of the
    list.
    
    @param dataset: AstroData object
    @type dataset:  <AstroData> instance

    @return: An astrodata id.
    @rtype: <str>

    """
    return dataset.data_label()

def generate_fringe_list_id(dataset):
    """
    Generate an id from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @return: A stackable id.
    @rtype: string
    """
    return generate_stackable_id(dataset)
