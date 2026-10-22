import astropy.modeling.polynomial as P

Model_dict = dict((T.__name__, T) for T in [P.Chebyshev2D, P.Chebyshev1D])

def serialize_poly_model(p):

    assert isinstance(p, P.PolynomialBase)

    if len(p.inputs) == 1:
        prefixes = [""]
    elif len(p.inputs) == 2:
        prefixes = ["{}_".format(_) for _ in p.inputs]
    else:
        raise ValueError("dimention larger than 2d is not supported. : {}".format(p.inputs))

    _serialized = {}
    for prefix in prefixes:
        for par in ["degree", "domain", "window"]:
            par_name = prefix + par
            v = getattr(p, par_name)
            _serialized[par_name] = v

    for par_name in ["param_names", "parameters"]:
        _serialized[par_name] = getattr(p, par_name)

    module_name = p.__class__.__module__
    klass_name = p.__class__.__name__

    return (module_name, klass_name, _serialized)


def deserialize_poly_model(module_name, klass_name, serialized):

    try:
        T = Model_dict[klass_name]
    except AttributeError:
        raise AttributeError("no poly model ({}) is defined".format(klass_name))

    assert issubclass(T, P.PolynomialBase)

    if hasattr(T, "n_inputs"): # newer astropy
        n_inputs = T.n_inputs
    else:  # for astropy < 4?
        n_inputs = len(T.inputs)

    if n_inputs == 1:
        prefixes = [""]  # FIXME: this might be ["x_"]. Need to be checked.
    elif n_inputs == 2:
        prefixes = ["x_", "y_"]
    else:
        raise ValueError(f"Unsupported Polynomial with n_inputs : {T.n_inputs}")

    serialized = serialized.copy()
    degrees = [serialized.pop(prefix+"degree") for prefix in prefixes]

    param_names = serialized.pop("param_names")
    parameters = serialized.pop("parameters")
    kwargs = dict(zip(param_names, parameters))
    kwargs.update(serialized)

    p = T(*degrees, **kwargs)

    return p

def test():
    
    p1 = P.Chebyshev2D(4, 3, c0_0=178.06976370696114, c1_0=1.2259796928216562, c2_0=-0.039693725997150944, c3_0=0.00025125807852843096, c4_0=0.00014173127277197645, c0_1=0.47279945290424674, c1_1=0.010547446717582997, c2_1=-0.00030744595719575896, c3_1=-0.0004721208753736342, c4_1=0.0005199111288754786, c0_2=0.01962380256315047, c1_2=-0.006113075901906929, c2_2=0.0006263988054950944, c3_2=-0.0005470157256276083, c4_2=0.0005138421175075688, c0_3=-0.0001892239475282825, c1_3=0.0002596208773213599, c2_3=0.00031317233972239596, c3_3=-0.0003102771114401838, c4_3=0.0003311264577003732)


    module_name, klass_name, serialized = serialize_poly_model(p1)

    p2 = deserialize_poly_model(module_name, klass_name, serialized)

    import numpy as np
    assert np.all(p1.parameters == p2.parameters)

if __name__ == "__main__":
    test()
