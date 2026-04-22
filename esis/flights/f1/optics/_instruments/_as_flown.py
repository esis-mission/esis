import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

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

    model = update_esis_model(guess, model, units)

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
    # print(f'{cc=}')



    # merit = lse + (distance_off_target)
    merit = -1000*cc + distance_off_target
    # print(f'{merit.ndarray=}',f'{guess=}')

    return merit.ndarray


def update_esis_model(guess, model, units):
    guess = [guess * unit for guess, unit in zip(guess, units)]

    g_yaw, g_pitch, g_roll, field_stop_roll, d_grating, primary_displacement, model_pitch, model_yaw, model_roll = guess

    model.grating.yaw = g_yaw
    model.grating.pitch = g_pitch
    model.grating.roll = g_roll
    model.field_stop.roll = field_stop_roll
    model.grating.rulings.spacing.coefficients[0] = d_grating
    model.primary_mirror.sag.focal_length = -1000 * u.mm + primary_displacement
    model.primary_mirror.translation.z = -primary_displacement
    model.pitch = model_pitch
    model.yaw = model_yaw
    model.roll = model_roll

    return model

from datetime import datetime
from pathlib import Path
class DECallback:
    def __init__(self, output_directory: Path):
        """
        output_directory: A pathlib.Path object pointing to the
                          already-created timestamped folder.
        """
        self.output_dir = output_directory

        # Define internal file paths using the passed directory
        self.data_log = self.output_dir / "convergence_data.csv"
        self.console_log = self.output_dir / "full_output.log"
        self.plot_path = self.output_dir / "convergence_plot.png"

        self.iteration = 0
        self.best_energies = []
        self.pop_std_history = []

        # Initialize logs with headers
        self.data_log.write_text("Iteration,Best_Energy,Pop_Std\n")
        self.console_log.write_text(f"--- Optimization Start: {datetime.now()} ---\n")

    def log_and_print(self, message):
        """Prints to console and appends to the log file."""
        print(message)
        with self.console_log.open("a") as f:
            f.write(message + "\n")

    def __call__(self, intermediate_result):
        self.iteration += 1
        xk = intermediate_result.x
        best_fun = intermediate_result.fun
        pop_std = np.std(intermediate_result.population_energies)

        self.best_energies.append(best_fun)
        self.pop_std_history.append(pop_std)

        # Append numerical data to CSV
        with self.data_log.open("a") as f:
            f.write(f"{self.iteration},{best_fun:.8e},{pop_std:.4e}\n")

        # Log parameters x and mirror console
        msg = (f"Iter {self.iteration:03d} | Energy: {best_fun:.6e} | "
               f"x: {np.array2string(xk, precision=4, separator=', ')}")
        self.log_and_print(msg)

        self.save_plot()

    def save_plot(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.array(self.best_energies)+1000, 'b-')
        plt.title("Best Fit Energy")
        plt.yscale('log')
        plt.subplot(1, 2, 2)
        plt.plot(self.pop_std_history, 'r-')
        plt.title("Population Diversity (Std Dev)")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

