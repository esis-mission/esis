import astropy.units as u

__all__ = [
    "wavelength_Ne_VII",
    "wavelength_Si_XII",
]

#: The wavelength of the Ne VII line in the ESIS-II passband
wavelength_Ne_VII = 465.221 * u.AA

#: The wavelength of the Si XII line in the ESIS-II passband
wavelength_Si_XII = 499.406 * u.AA
