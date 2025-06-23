# rotate_alm_helper.py
import healpy as hp
import numpy as np

def healpix_lmax_from_C(C):
    """Inverse of  C = (L+1)(L+2)//2  (already in your loading.py)."""
    L = int(np.floor((-3 + np.sqrt(1 + 8*C)) / 2))
    if (L+1)*(L+2)//2 != C:
        raise ValueError("invalid Alm length")
    return L


def rotate_trajectory(arr, psi, theta, phi):
    """
    Rotate an entire trajectory of HEALPix Alms.

    Parameters
    ----------
    arr : np.ndarray, shape (T, C, M), complex64
          m≥0 compact storage, band-ordered (HEALPix convention)
    psi, theta, phi : floats
          Z–Y–Z Euler angles **in radians**

    Returns
    -------
    out : np.ndarray, same shape / dtype as arr
    """
    T, C, M = arr.shape
    Lmax    = healpix_lmax_from_C(C)

    # build a single Rotator once
    rot = hp.rotator.Rotator(rot=[phi, theta, psi],      # ZYZ ⇒ intrinsic
                             deg=False,                  # already radians
                             eulertype='ZYZ')

    out = np.empty_like(arr)
    for m in range(M):
        for t in range(T):
            alm_in  = arr[t, :, m].astype(np.complex128)   # HP needs 128-bit
            alm_out = rot.rotate_alm(alm_in, lmax=Lmax)    # returns new array
            out[t, :, m] = alm_out.astype(np.complex64)

    return out

