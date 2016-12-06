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

def generate_fringe_list_id(dataset):
    """
    Generate an id from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @return: A stackable id.
    @rtype: string

    """
    return generate_stackable_id(dataset)
