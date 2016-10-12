#
#                                                                   IDFactory.py
#------------------------------------------------------------------------------ 
import hashlib

from astrodata import AstroData

#------------------------------------------------------------------------------ 
def generate_md5_file(filename):
    block_size = 1 << 18        # 262 144 byte chunks
    md5 = hashlib.md5()
    with open(filename) as f:
        data = f.read(block_size)
        md5.update(data)

    return md5.hexdigest()
    
def generate_stackable_id(dataset):
    """
    Generate an ID from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @return: A stackable id.
    @rtype: string  

    """
    _v = "1_0"
    if isinstance(dataset, str):
        ad = AstroData(dataset)
        ID = "{}{}".format(_v, str(ad.group_id()))
    elif isinstance(dataset, AstroData):
        ID = "{}{}".format(_v, str(dataset.group_id()))
    else:
        raise TypeError("BAD ARGUMENT: {}".format(dataset))

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
    as you will have (DATALAB, bias). So, any steps inbetween getProcessedBias and
    biasCorrect will have no impact.
    
    2) Fringe:
    
    Fringe uses this as a FringeID, which is based off the first input of the list.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str

    @return: An astrodata id.
    @rtype: string

    """
    if isinstance(dataset, str):
        ad = AstroData(dataset)
        ADID = ad.data_label().as_pytype()
        return ADID
    elif isinstance(dataset, AstroData):
        ADID = dataset.data_label().as_pytype()
        return ADID
    else:
        raise TypeError("BAD ARGUMENT: {}".format(type(dataset)))
    
def generate_fringe_list_id(dataset):
    '''
    Generate an id from which all similar stackable data will have in common.
    
    @param dataset: Input AstroData instance or fits filename.
    @type dataset: AstroData instances or str
    
    @return: A stackable id.
    @rtype: string
    '''
    return generate_stackable_id(dataset)

def make_safe_id(spaced_id):
    return spaced_id.replace(" ", "_")

