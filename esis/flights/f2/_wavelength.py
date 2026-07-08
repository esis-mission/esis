import astropy.units as u

__all__ = [
    "wavelength_Ne_VII",
    "wavelength_Si_XII",
    "wavelength_HeNe",
]

#: The wavelength of the Ne VII line in the ESIS-II passband
wavelength_Ne_VII = 465.221 * u.AA

#: The wavelength of the Si XII line in the ESIS-II passband
wavelength_Si_XII = 499.406 * u.AA

#: The wavelength of the HeNe laser used with the visible alignment gratings
wavelength_HeNe = 632.8 * u.nm
