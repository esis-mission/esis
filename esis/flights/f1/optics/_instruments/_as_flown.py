import numpy as np
import astropy.units as u

def esis_distortion_merit(
        guess,
        model=None,
        units=None,
        esis_img=None,
        scene=None,
        pupil=None,
        axis_wavelength=None,
        axis_field=None,
):

    guess = [guess * unit for guess, unit in zip(guess, units)]

    # g_yaw, g_pitch, g_roll, camera_phi = guess
    g_yaw, g_pitch, g_roll, camera_roll, d_grating, primary_displacement, model_pitch, model_yaw, model_roll = guess

    model.grating.yaw = g_yaw
    model.grating.pitch = g_pitch
    model.grating.roll = g_roll
    # model.camera.sensor.azimuth = camera_phi
    # model.filter.azimuth = camera_phi
    model.camera.sensor.roll = camera_roll
    model.grating.rulings.spacing.coefficients[0] = d_grating

    model.primary_mirror.sag.focal_length = -1000 * u.mm + primary_displacement
    model.primary_mirror.translation.z = -primary_displacement
    model.pitch = model_pitch
    model.yaw = model_yaw
    model.roll = model_roll

    if model.system:
        del model.system
    image = model.system.image(
        scene=scene,
        pupil=pupil,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
        noise=False,
    )

    position = model.system.rayfunction_default.outputs.position.to(u.mm)

    # fit_img = image.outputs.sum(axis=("spectral_line", "tilt", "wavelength")).value
    fit_img = image.outputs.sum(axis=("spectral_line", "wavelength")).value

    # merit = ne.numexpr.evaluate('sum((fit_img - esis_img)*(fit_img - esis_img))')
    distance_off_target = np.square(position.mean().length.value)
    # print(f'{distance_off_target=}')

    # lse = 1E-15 * np.sqrt(np.sum(np.square(fit_img - esis_img)))
    # print(f'{lse=}')

    esis_img = esis_img-esis_img.mean()
    esis_img = esis_img/esis_img.std()

    fit_img = fit_img-fit_img.mean()
    if fit_img.ndarray.std() !=0:
        fit_img = fit_img/fit_img.std()

    cc = (fit_img*esis_img).sum()
    cc = cc/fit_img.size
    print(f'{cc=}')



    # merit = lse + (distance_off_target)
    merit = -1000*np.abs(cc) + (distance_off_target)
    print(f'{merit.ndarray=}',f'{guess=}')

    return merit.ndarray
