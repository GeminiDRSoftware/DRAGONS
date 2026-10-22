import numpy as np

# power spectrum


def _get_amp_wise_rfft(d):
    dr = d.reshape((-1, 64, 2048))
    m = np.median(dr, axis=1)

    return np.fft.rfft(m, axis=1)


def get_amp_wise_noise_spectrum(cube):
    return _get_amp_wise_rfft(cube)


def _get_c64_wise_rfft(d):
    dr = d.reshape((-1, 2048, 32, 64))
    m = np.median(dr, axis=3)

    return np.fft.rfft(m, axis=1)


def get_c64_wise_noise_spectrum(cube):
    return _get_c64_wise_rfft(cube)


# def _remove_kk(d):
#     kk = ["p64_per_column", "row_wise_bias"]
#     d1 = apply_pipes(d, [pipes[k] for k in kk])

#     return d1


def get_amp_wise_real(d):
    dr = d.reshape((-1, 64, 2048))
    m = np.median(dr, axis=1)

    return m


def get_amp_wise_stacked(d):
    dr = d.reshape((-1, 64, 2048))
    return dr


def get_amp_wise_rfft(d):
    dr = d.reshape((-1, 64, 2048))
    m = np.median(dr, axis=1)

    return np.fft.rfft(m, axis=1)


def make_model_from_rfft(q, kslice):
    orig_shape = q.shape
    qr = q.reshape((-1,) + orig_shape[-1:])
    q0 = np.zeros_like(qr)
    q0[:, kslice] = qr[:, kslice]

    return np.fft.irfft(q0, axis=-1).reshape(orig_shape[:-1] + (-1,))

# def make_model_from_rfft(q, kslice):
#     q0 = np.zeros_like(q)
#     q0[:, kslice] = q[:, kslice]

#     return np.fft.irfft(q0, axis=1)
