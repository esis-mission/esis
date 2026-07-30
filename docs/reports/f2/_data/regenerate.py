"""Regenerate ``dem.npz`` and ``gofnt.ecsv`` for the synthetic-image report.

This script is not executed at documentation build time: it needs heavyweight
dependencies (``sunpy``, ``aiapy``, ``reproject``, ``demregpy``, ``fiasco`` and
the ~2 GB CHIANTI database) and network access to JSOC.  Run it manually with

.. code-block:: bash

    pip install sunpy aiapy reproject demregpy fiasco
    python regenerate.py

The first invocation downloads the six AIA images via ``sdo`` and the CHIANTI
database via ``fiasco``; both are cached for subsequent runs.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pathlib

import astropy.time
import astropy.units as u
import numpy as np

TIME = "2016-07-15T00:00:06"
CENTER = (-30.0, 70.0) * u.arcsec  # ESIS-II boresight, helioprojective
HALF_DEM = 520 * u.arcsec  # half-width of the DEM inversion box
HALF_SCENE = 250.0  # arcsec, half-width of the committed scene
BAND = (430, 535) * u.AA
DENSITY = 1e9 * u.cm**-3
T_EDGES_LOG = np.arange(5.6, 7.21, 0.1)
THRESHOLD_FLUX = 0.005  # keep lines above this fraction of the band photon flux

IONS = [
    "O III",
    "O IV",
    "O V",
    "O VI",
    "Ne IV",
    "Ne V",
    "Ne VI",
    "Ne VII",
    "Ne VIII",
    "Na IX",
    "Mg VI",
    "Mg VII",
    "Mg VIII",
    "Mg IX",
    "Mg X",
    "Al IX",
    "Al X",
    "Al XI",
    "Si VIII",
    "Si IX",
    "Si X",
    "Si XI",
    "Si XII",
    "S IX",
    "S X",
    "S XI",
    "S XII",
    "S XIII",
    "Ar XI",
    "Ar XII",
    "Ar XIII",
    "Ca IX",
    "Ca X",
    "Fe IX",
    "Fe X",
    "Fe XI",
    "Fe XII",
    "Fe XIII",
    "Fe XIV",
    "Fe XV",
    "Fe XVI",
]

directory = pathlib.Path(__file__).parent


def dem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert the DEM from the six optically-thin AIA EUV channels."""
    import aiapy.calibrate
    import demregpy
    import named_arrays as na
    import sdo.aia
    import sunpy.map
    from astropy.coordinates import SkyCoord

    wavelengths = [94, 131, 171, 193, 211, 335]

    files = sdo.aia.prep(
        sdo.aia.download(
            sdo.aia.urls(
                time_start="2016-07-15T00:00:00",
                time_stop="2016-07-15T00:00:12",
                wavelength=na.ScalarArray(wavelengths * u.AA, axes="wavelength"),
            )
        )
    )
    files = files[dict(time=0)]

    maps = {
        w: sunpy.map.Map(str(files[dict(wavelength=i)].ndarray))
        for i, w in enumerate(wavelengths)
    }

    ref = maps[193]
    bl = SkyCoord(*(CENTER - HALF_DEM), frame=ref.coordinate_frame)
    tr = SkyCoord(*(CENTER + HALF_DEM), frame=ref.coordinate_frame)
    ref_cut = ref.submap(bl, top_right=tr)

    nf = len(maps)
    ny, nx = ref_cut.data.shape
    dn = np.empty((ny, nx, nf))
    texp = np.empty(nf)

    for k, (w, m) in enumerate(maps.items()):
        degradation = aiapy.calibrate.degradation(
            channel=w * u.AA,
            obstime=astropy.time.Time(TIME),
        )
        degradation = np.atleast_1d(degradation.to_value(u.one)).squeeze()[()]
        cut = ref_cut if m is ref else m.reproject_to(ref_cut.wcs)
        dn[..., k] = cut.data / degradation
        texp[k] = m.exposure_time.to_value(u.s)

    # 2x2 binning to 1.2 arcsec pixels
    ny2, nx2 = ny // 2, nx // 2
    dn = dn[: ny2 * 2, : nx2 * 2].reshape(ny2, 2, nx2, 2, nf).mean(axis=(1, 3))
    dn = np.clip(np.nan_to_num(dn), 0, None)
    npix = 4

    dn_s = dn / texp

    # noise model following the demreg AIA examples
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    dn2ph = gains * np.array(wavelengths) / 3397.0
    shot = np.sqrt(dn * dn2ph * npix) / dn2ph / npix
    edn = np.sqrt(shot**2 + 1.15**2 / npix) / texp
    edn = np.maximum(edn, 1e-2)

    channels, tresp_logt, trmatrix = demregpy.load_aia_response()
    temps = 10.0**T_EDGES_LOG

    nt = len(temps) - 1
    result = np.empty((ny2, nx2, nt))
    block = 128
    for j0 in range(0, ny2, block):
        j1 = min(j0 + block, ny2)
        d, *_ = demregpy.dn2dem(
            dn_s[j0:j1],
            edn[j0:j1],
            trmatrix,
            tresp_logt,
            temps,
            max_iter=15,
        )
        result[j0:j1] = d
        print(f"DEM rows {j0}:{j1} done", flush=True)

    # scene coordinates relative to the boresight
    scale = ref_cut.scale.axis1.to_value(u.arcsec / u.pix) * 2
    x_vertices = (np.arange(nx2 + 1) - nx2 / 2) * scale
    y_vertices = (np.arange(ny2 + 1) - ny2 / 2) * scale

    return result, x_vertices, y_vertices


def gofnt(dem_cube: np.ndarray):
    """Compute G(T) for every line in the passband and keep the bright ones."""
    import astropy.constants
    import fiasco
    from astropy.table import Table

    logt = 0.5 * (T_EDGES_LOG[:-1] + T_EDGES_LOG[1:])
    temperature = 10.0**logt * u.K

    names, wavelengths, matrices = [], [], []
    for ion_name in IONS:
        try:
            ion = fiasco.Ion(ion_name, temperature)
            g = ion.contribution_function(DENSITY)[:, 0, :]
            wl = ion.transitions.wavelength[ion.transitions.is_bound_bound]
        except Exception as error:
            print(f"{ion_name}: skipped ({error})")
            continue
        keep = (wl >= BAND[0]) & (wl <= BAND[1])
        for i in np.where(keep)[0]:
            names.append(ion_name)
            wavelengths.append(wl[i].to_value(u.AA))
            matrices.append(g[:, i].to_value(u.erg * u.cm**3 / u.s))
        print(f"{ion_name}: {int(keep.sum())} lines in band")

    g_matrix = np.array(matrices)

    # rank by DEM-weighted photon flux over the scene
    dT = np.diff(10.0**T_EDGES_LOG)
    em_mean = (np.clip(dem_cube, 0, None) * dT).mean(axis=(0, 1))
    intensity = 0.83 / (4 * np.pi) * g_matrix @ em_mean
    energy = (
        astropy.constants.h * astropy.constants.c / (np.array(wavelengths) * u.AA)
    ).to_value(u.erg)
    flux = intensity / energy
    keep = flux / flux.sum() > THRESHOLD_FLUX
    print(f"keeping {keep.sum()} of {len(flux)} lines")

    table = Table(
        {
            "ion": np.array(names)[keep],
            "wavelength": np.array(wavelengths)[keep] * u.AA,
            "gofnt": g_matrix[keep] * u.erg * u.cm**3 / u.s,
        },
    )
    table.meta["logt"] = list(np.round(logt, 3))
    table.meta["density"] = "1e9 cm-3"
    table.meta["abundance"] = "sun_coronal_1992_feldman_ext"
    table.meta["database"] = "CHIANTI (via fiasco)"
    table.meta["description"] = (
        "Contribution functions G(n_e, T) for the lines in the ESIS-II passband "
        "contributing more than 0.5% of the DEM-weighted photon flux of the "
        "2016-07-15 00:00 UT active-region scene."
    )
    table.sort("wavelength")
    return table


def main():
    """Regenerate ``dem.npz`` and ``gofnt.ecsv``."""
    dem_cube, x_vertices, y_vertices = dem()

    table = gofnt(dem_cube)
    table.write(directory / "gofnt.ecsv", format="ascii.ecsv", overwrite=True)

    keep_x = (x_vertices[:-1] > -HALF_SCENE) & (x_vertices[1:] < HALF_SCENE)
    keep_y = (y_vertices[:-1] > -HALF_SCENE) & (y_vertices[1:] < HALF_SCENE)
    ix = np.where(keep_x)[0]
    iy = np.where(keep_y)[0]

    np.savez_compressed(
        directory / "dem.npz",
        dem=np.clip(dem_cube[iy[0] : iy[-1] + 1, ix[0] : ix[-1] + 1], 0, None).astype(
            np.float32
        ),
        x_vertices_arcsec=x_vertices[ix[0] : ix[-1] + 2].astype(np.float32),
        y_vertices_arcsec=y_vertices[iy[0] : iy[-1] + 2].astype(np.float32),
        t_edges_log=T_EDGES_LOG,
        description=np.array(
            "Regularized DEM (cm-5 K-1) of the AR 12565/12567 pair from the six "
            "optically-thin SDO/AIA EUV channels (94/131/171/193/211/335 A, no "
            "304) at 2016-07-15 00:00 UT, inverted with demregpy against the SSW "
            "evenorm temperature response (CHIANTI, Feldman 1992 extended "
            "coronal abundances). Grid: 1.2 arcsec pixels; coordinates are "
            "helioprojective offsets from the assumed ESIS-II boresight at "
            "(Tx, Ty) = (-30, +70) arcsec."
        ),
    )
    print("wrote", directory / "dem.npz", "and", directory / "gofnt.ecsv")


if __name__ == "__main__":
    main()
