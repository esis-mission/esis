"""O IV 608.40 / 609.83 photon ratio vs electron density (CHIANTI via fiasco)."""

import astropy.units as u
import numpy as np
import fiasco

temperature = np.geomspace(1e4, 1e7, 61) * u.K

for log_n in [8, 9, 10, 11, 12]:
    density = 10.0**log_n * u.cm**-3
    ion = fiasco.Ion("O 4", temperature)
    contribution = ion.contribution_function(density)
    wavelength = ion.transitions.wavelength[~ion.transitions.is_twophoton]
    idx_608 = np.argmin(np.abs(wavelength - 608.40 * u.AA))
    idx_610 = np.argmin(np.abs(wavelength - 609.83 * u.AA))
    g_608 = contribution[:, 0, idx_608]
    g_610 = contribution[:, 0, idx_610]
    i_peak = int(np.argmax(g_610.value))
    t_peak = temperature[i_peak]
    # contribution_function is in energy units; convert to photon ratio
    ratio_energy = (g_608 / g_610)[i_peak]
    ratio_photon = float(ratio_energy) * (608.40 / 609.83) ** -1 * (609.83 / 608.40) ** 0
    # photon ratio = (energy ratio) * (lambda_608 / lambda_610)
    ratio_photon = float(ratio_energy) * (608.40 / 609.83)
    print(
        f"n_e = 1e{log_n} cm^-3:  T_peak = {t_peak:.2e}  "
        f"energy ratio 608/610 = {float(ratio_energy):.4f}  "
        f"photon ratio = {ratio_photon:.4f}",
        flush=True,
    )
    if log_n == 8:
        print("  wavelengths matched:", wavelength[idx_608], wavelength[idx_610])
        # temperature dependence at this density
        for i in range(0, len(temperature), 6):
            if g_610[i].value > 0:
                print(
                    f"    T={temperature[i]:.2e}: ratio "
                    f"{float((g_608[i] / g_610[i])):.4f}"
                )
