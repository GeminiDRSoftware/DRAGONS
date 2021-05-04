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

            new_kwargs = {p.name: p.default
                          for p in inspect.signature(fn).parameters.values()
                          if p.default is not p.empty}
            new_kwargs.update(**kwargs)
            for k, v in new_kwargs.items():
                if k in all_descriptors and v is None:
                    desc_kwargs = DESCRIPTOR_KWARGS.get(k, {})
                    # Because we can't expect other people to use the IRAF system
                    if k == "dispersion_axis":
                        new_kwargs[k] = len(ext.shape) - ext.dispersion_axis()
                    else:
                        new_kwargs[k] = getattr(ext, k)(**desc_kwargs)
            return fn(ext, *args, **new_kwargs)
        return gn
    return inner_decorator
