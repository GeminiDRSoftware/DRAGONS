from functools import wraps
import inspect


def insert_descriptor_values(*descriptors):
    """
    Decorates a function which operates on an NDData-like object so that
    it can be sent an NDAstroData object and descriptor values will be
    inserted into the kwargs if the default is None and no value is
    explicitly provided.

    If a list of descriptors is provided, then only those in this list
    will be inserted into the kwargs.
    """
    def inner_decorator(fn):
        DESCRIPTOR_KWARGS = {"central_wavelength": {"asNanometers": True},
                             "dispersion": {"asNanometers": True}
                            }
        @wraps(fn)
        def gn(ext, *args, **kwargs):
            try:
                all_descriptors = ext.descriptors
            except AttributeError:
                return fn(ext, *args, **kwargs)
            if descriptors:
                all_descriptors = descriptors

            fn_kwargs = {p.name: p.default
                         for p in inspect.signature(fn).parameters.values()
                         if p.default is not p.empty}
            for k, v in fn_kwargs.items():
                if k in all_descriptors and v is None:
                    desc_kwargs = DESCRIPTOR_KWARGS.get(k, {})
                    # Because we can't expect other people to use the IRAF system
                    if k == "dispersion_axis":
                        kwargs[k] = len(ext.shape) - ext.dispersion_axis()
                    else:
                        kwargs[k] = getattr(ext, k)(**desc_kwargs)
            return fn(ext, *args, **kwargs)
        return gn
    return inner_decorator


def unpack_nddata(fn):
    """
    This decorator wraps a function/staticmethod that expects separate
    data, mask, and variance parameters and allows an NDAstroData instance
    to be sent instead. This is similar to nddata.support_nddata, but
    handles variance and doesn't give warnings if the NDData instance has
    attributes set which aren't picked up by the function.

    It's also now happy with an np.ma.masked_array
    """
    @wraps(fn)
    def wrapper(data, *args, **kwargs):
        if hasattr(data, 'mask'):
            if 'mask' not in kwargs:
                kwargs['mask'] = data.mask
            if ('variance' in inspect.signature(fn).parameters and
                    'variance' not in kwargs and hasattr(data, 'variance')):
                kwargs['variance'] = data.variance
            ret_value = fn(data.data, *args, **kwargs)
        else:
            ret_value = fn(data, *args, **kwargs)
        return ret_value
    return wrapper
