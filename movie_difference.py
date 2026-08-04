#!/usr/bin/env python3
"""
Make a difference movie of two channels projected onto the sky, O V only.

Channel 1 (the coalignment anchor) minus channel 2, frame by frame through
the flight, with and without the fitted correction applied to channel 2.

The channels have different gains, so each sky-plane image is standardized
before differencing: otherwise the difference is dominated by one channel
simply being brighter, and the misalignment is invisible underneath it.
What a residual misalignment looks like in such a difference is a
bright/dark dipole hugging every structure edge, which reverses sense
across the field of view when the error has a gradient.
"""

import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation

import astropy.units as u
import astropy.constants
import astropy.table
import named_arrays as na

import esis
import coalign

DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
OTHER = int(os.environ.get("ESIS_OTHER", "2"))
LINE = "O V"


def main() -> None:
    """Render the difference movie with and without the correction."""
    directory = pathlib.Path(__file__).parent / "coalignment_20260804"
    table = astropy.table.QTable.read(
        directory / "coalignment.ecsv",
        format="ascii.ecsv",
    )
    correction = {int(row["channel"]): row for row in table}

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system

    l1 = esis.flights.f1.data.level_1()
    num_time = na.shape(l1.outputs)["time"]
    num_channel = na.shape(l1.outputs)["channel"]
    print(f"{num_time} frames, {num_channel} channels", flush=True)

    wavelength_line = na.stack(list(coalign.LINES.values()), axis="line")
    velocity = na.linspace(-100, 100, axis="wavelength", num=3) * u.km / u.s
    wavelength = (wavelength_line * (1 + velocity / astropy.constants.c)).to(u.AA)
    wavelength = wavelength.combine_axes(
        axes=("line", "wavelength"),
        axis_new="wavelength",
    )

    print("linearizing...", flush=True)
    linear = system.linearize(wavelength=wavelength, degree=DEGREE)
    distortion = linear.distortion

    sky = coalign.sky_grid(system, NUM_SKY)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    pixel_x = span_x / NUM_SKY
    pixel_y = span_y / NUM_SKY
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2

    names = list(coalign.LINES)
    wavelength_rest = coalign.LINES[LINE]
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(
        coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean
    )
    dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half

    row = correction[OTHER]
    cx = 2 * (sky.x - center_x) / span_x
    cy = 2 * (sky.y - center_y) / span_y
    dx = row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
    dy = row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw
    warp = {
        OTHER: na.Cartesian2dVectorArray(
            x=sky.x - dx * pixel_x,
            y=sky.y - dy * pixel_y,
        )
    }

    def standardize(a: np.ndarray) -> np.ndarray:
        """Remove the gain and offset of a channel, keeping its structure."""
        valid = np.isfinite(a)
        if valid.sum() < 10:
            return np.full_like(a, np.nan)
        return (a - np.nanmean(a)) / max(np.nanstd(a), 1e-12)

    frames = []
    for t in range(num_time):
        image = na.value(l1.outputs[dict(time=t)])

        panels = []
        for label, w in (("uncorrected", None), ("corrected", warp)):
            sky_images = coalign.project(
                distortion=distortion,
                image=image,
                sky=sky,
                wavelength=wavelength_rest,
                num_channel=num_channel,
                warp=w,
            )
            a = standardize(sky_images[coalign.ANCHOR])
            b = standardize(sky_images[OTHER])
            panels.append(a - b)

        frames.append(panels)
        rms = [float(np.sqrt(np.nanmean(p**2))) for p in panels]
        print(f"  frame {t:2d}: rms difference uncorrected {rms[0]:.4f}, "
              f"corrected {rms[1]:.4f} ({100 * (1 - rms[1] / rms[0]):+.1f}%)",
              flush=True)

    everything = np.concatenate([np.ravel(p) for f in frames for p in f])
    limit = float(np.nanpercentile(np.abs(everything), 99))
    print(f"\ncolour limit +/-{limit:.3f}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
    images = []
    for ax, label in zip(axs, ("uncorrected", "corrected")):
        im = ax.imshow(
            frames[0][0 if label == "uncorrected" else 1].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=+limit,
        )
        ax.set_title(f"channel {coalign.ANCHOR} - channel {OTHER}, {label}")
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(im)
    title = fig.suptitle("")
    fig.colorbar(images[1], ax=axs, shrink=0.8, label="standardized difference")

    def update(t):
        for i, im in enumerate(images):
            im.set_data(frames[t][i].T)
        title.set_text(f"ESIS {LINE} sky-plane difference, frame {t}")
        return images

    animation = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=num_time,
        interval=250,
        blit=False,
    )

    path = directory / f"difference_{LINE.replace(' ', '')}_ch{coalign.ANCHOR}_ch{OTHER}.gif"
    animation.save(path, writer=matplotlib.animation.PillowWriter(fps=4), dpi=80)
    print(f"wrote {path}", flush=True)

    # a still of the median frame, easier to inspect closely than the movie
    t = num_time // 2
    for i, label in enumerate(("uncorrected", "corrected")):
        images[i].set_data(frames[t][i].T)
    title.set_text(f"ESIS {LINE} sky-plane difference, frame {t}")
    path_still = path.with_suffix(".png")
    fig.savefig(path_still, dpi=110)
    print(f"wrote {path_still}", flush=True)


if __name__ == "__main__":
    main()
