import astrodata
import gemini_instruments

import numpy as np

from gempy.utils import logutils
from geminidr.gemini.lookups.timestamp_keywords import timestamp_keys

def ad_compare(ad1, ad2):
    """
    Compares the tags, headers, and pixel values of two images

    Parameters
    ----------
    ad1: AstroData/other
        first file (AD or can be opened by astrodata.open())
    ad2: AstroData/other
        second file (AD or...)

    Returns
    -------

    """
    log = logutils.get_logger(__name__)

    if not isinstance(ad1, astrodata.AstroData):
        ad1 = astrodata.open(ad1)
    if not isinstance(ad2, astrodata.AstroData):
        ad2 = astrodata.open(ad2)

    fname1 = ad1.filename
    fname2 = ad2.filename
    ok = True

    # If images have different lengths, give up now
    if len(ad1) != len(ad2):
        log.warning('Files have different numbers of extensions: {} v {}'.
                      format(len(ad1), len(ad2)))
        return False

    # Check tags
    if ad1.tags == ad2.tags:
        log.stdinfo('TAGS match')
    else:
        log.warning('TAGS do not match:')
        log.warning('  {} (1): {}'.format(fname1, ad1.tags))
        log.warning('  {} (2): {}'.format(fname2, ad2.tags))
        ok = False

    # Check header keywords in PHU and all extension HDUs
    log.stdinfo('Checking headers...')
    for i, (h1, h2) in enumerate(zip(ad1.header, ad2.header)):
        hstr = 'PHU' if i==0 else 'HDU {}'.format(i)
        log.stdinfo('  Checking {}'.format(hstr))

        # Compare keyword lists
        s1 = set(h1.keys()) - set(['HISTORY', 'COMMENT'])
        s2 = set(h2.keys()) - set(['HISTORY', 'COMMENT'])
        if s1 != s2:
            log.warning('Header keyword mismatch...')
            if s1-s2:
                log.warning('  {} (1) contains keywords {}'.
                            format(fname1, s1-s2))
            if s2-s1:
                log.warning('  {} (2) contains keywords {}'.
                            format(fname2, s2-s1))
            ok = False

        # Compare values for meaningful keywords
        for kw in h1:
            # GEM-TLM is "time last modified"
            if kw not in timestamp_keys.values() and kw not in ['GEM-TLM',
                                                    'HISTORY', 'COMMENT']:
                try:
                    v1, v2 = h1[kw], h2[kw]
                except KeyError:  # Missing keyword in AD2
                    continue
                if isinstance(v1, float):
                    if abs(v1 - v2) > 0.01:
                        log.warning('{} value mismatch: {} v {}'.
                                    format(kw, v1, v2))
                        ok = False
                else:
                    if v1 != v2:
                        log.warning('{} value mismatch: {} v {}'.
                                    format(kw, v1, v2))
                        ok = False

    # Check REFCAT status, just equal lengths
    attr1 = getattr(ad1, 'REFCAT', None)
    attr2 = getattr(ad2, 'REFCAT', None)
    if (attr1 is None) ^ (attr2 is None):
        log.warning('    Attribute mismatch for REFCAT: {} v {}'.
                    format(attr1 is not None, attr2 is not None))
        ok = False
    elif attr1 is not None and attr2 is not None:
        if len(attr1) != len(attr2):
            log.warning('    REFCAT lengths differ: {} v {}'.
                        format(len(attr1), len(attr2)))
            ok = False

    # Extension by extension, check all the attributes
    log.stdinfo('Checking extensions...')
    for ext1, ext2 in zip(ad1, ad2):
        log.stdinfo('  Checking extver {}'.format(ext1.hdr.EXTVER))
        for attr in ['data', 'mask', 'variance', 'OBJMASK', 'OBJCAT']:
            attr1 = getattr(ext1, attr, None)
            attr2 = getattr(ext2, attr, None)
            if (attr1 is None) ^ (attr2 is None):
                log.warning('    Attribute mismatch for {}: {} v {}'.
                            format(attr, attr1 is not None, attr2 is not None))
                ok = False
                continue
            if attr1 is not None and attr2 is not None:
                if attr == 'OBJCAT':
                    if len(attr1) != len(attr2):
                        log.warning('    OBJCAT lengths differ: {} v {}'.
                                    format(len(attr1), len(attr2)))
                        ok = False
                else:
                    # Pixel-data extensions
                    if attr1.dtype != attr2.dtype:
                        log.warning('    Datatype mismatch for {}: {} v {}'.
                                    format(attr, attr1.dtype, attr2.dtype))
                        ok = False
                    elif attr1.shape != attr2.shape:
                        log.warning('    Shape mismatch for {}: {} v {}'.
                                    format(attr, attr1.shape, attr2.shape))
                        ok = False
                    else:
                        diff = attr1 - attr2
                        maxdiff = np.max(abs(diff))
                        # Let's assume int arrays should be identical, but
                        # allow tolerance for float arrays.
                        # TODO: Maybe compare data difference against variance?
                        if 'int' in attr1.dtype.name:
                            if maxdiff > 0:
                                log.warning('    int arrays not identical: '
                                    'max difference {}'.format(maxdiff))
                                ok = False
                        elif maxdiff > 0.1:
                            log.warning('    float arrays differ: max difference '
                                        '{}'.format(maxdiff))
                            ok = False
    return ok