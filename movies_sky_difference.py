#!/usr/bin/env python3
"""
Movies of the inter-channel disagreement over the flight, with and without.

This is the part of the comparison that needs no inversion. Each channel's
Level-1 frame is projected onto a common sky grid and differenced against
the anchor, uncorrected and corrected, for every frame and both bright
lines. What a residual misalignment looks like in such a difference is a
bright/dark dipole hugging every structure edge, reversing sense across
the field when the error has a gradient — so the movies show directly
whether the correction removes the disagreement everywhere and at all
times, or only where and when it was fitted.

The channels have different gains, so each sky-plane image is standardized
before differencing: otherwise the difference is dominated by one channel
simply being brighter and the misalignment is invisible underneath it.

The sky-to-sensor mapping does not depend on the frame, so it is built once
per channel, line and arm and reused down the flight — the projection is
otherwise the entire cost.

Companion to `gpu/compare_movies.py`, which does the same for the inverted
Doppler maps and the MART residuals and needs the cluster.
"""

import os
import pathlib

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants
import astropy.table
import named_arrays as na

import esis
import coalign

DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
INTERPOLATION = os.environ.get("ESIS_INTERPOLATION", "linear")
FPS = int(os.environ.get("ESIS_FPS", "4"))


def standardize(a: np.ndarray) -> np.ndarray:
    """Remove the gain and offset of a channel, keeping its structure."""
    if np.isfinite(a).sum() < 10:
        return np.full_like(a, np.nan)
    return (a - np.nanmean(a)) / max(np.nanstd(a), 1e-12)


def main() -> None:
    """Render every difference movie and the summary of their rms."""
    directory = pathlib.Path(__file__).parent / "coalignment_20260804"
    out = pathlib.Path(__file__).parent / "docs/reports/channel_coalignment/movies_sky"
    out.mkdir(parents=True, exist_ok=True)

    table = astropy.table.QTable.read(
        directory / "coalignment.ecsv", format="ascii.ecsv"
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
    distortion = system.linearize(wavelength=wavelength, degree=DEGREE).distortion

    sky = coalign.sky_grid(system, NUM_SKY)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    pixel_x, pixel_y = span_x / NUM_SKY, span_y / NUM_SKY
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean)

    def warp_for(c: int, wavelength_rest: u.Quantity):
        """Corrected sky coordinates of one channel at one line."""
        row = correction[c]
        cx = 2 * (sky.x - center_x) / span_x
        cy = 2 * (sky.y - center_y) / span_y
        dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half
        dx = row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
        dy = row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw
        return na.Cartesian2dVectorArray(
            x=sky.x - dx * pixel_x,
            y=sky.y - dy * pixel_y,
        )

    others = [c for c in range(num_channel) if c != coalign.ANCHOR]
    summary = {}

    for name, wavelength_rest in coalign.LINES.items():
        # the mapping is frame-independent, so build it once per arm
        maps = {
            "uncorrected": coalign.sensor_coordinates(
                distortion, sky, wavelength_rest, num_channel
            ),
            "corrected": coalign.sensor_coordinates(
                distortion,
                sky,
                wavelength_rest,
                num_channel,
                warp={c: warp_for(c, wavelength_rest) for c in others},
            ),
        }

        differences = {c: {k: [] for k in maps} for c in others}
        for t in range(num_time):
            image = na.value(l1.outputs[dict(time=t)])
            for arm, coordinates in maps.items():
                anchor = standardize(
                    coalign.sample(
                        image,
                        coalign.ANCHOR,
                        *coordinates[coalign.ANCHOR],
                        interpolation=INTERPOLATION,
                    )
                )
                for c in others:
                    other = standardize(
                        coalign.sample(
                            image, c, *coordinates[c], interpolation=INTERPOLATION
                        )
                    )
                    differences[c][arm].append(anchor - other)
            print(f"  {name} frame {t:2d} projected", flush=True)

        for c in others:
            stack = {arm: np.stack(frames) for arm, frames in differences[c].items()}
            rms = {
                arm: np.sqrt(np.nanmean(a**2, axis=(1, 2))) for arm, a in stack.items()
            }
            summary[(name, c)] = rms
            change = 100 * (1 - rms["corrected"].mean() / rms["uncorrected"].mean())
            print(
                f"{name} channel {c}: mean rms difference"
                f" {rms['uncorrected'].mean():.4f} -> {rms['corrected'].mean():.4f}"
                f" ({change:+.1f}%)",
                flush=True,
            )

            limit = float(np.nanpercentile(np.abs(stack["uncorrected"]), 99))
            improvement = np.abs(stack["corrected"]) - np.abs(stack["uncorrected"])

            fig, axs = plt.subplots(1, 3, figsize=(14.0, 5.2), constrained_layout=True)
            panels = (stack["uncorrected"], stack["corrected"], improvement)
            titles = (
                "uncorrected",
                "coaligned",
                "|coaligned| − |uncorrected|",
            )
            limits = (limit, limit, limit / 2)
            images = []
            for ax, data, title, lim in zip(axs, panels, titles, limits):
                images.append(
                    ax.imshow(
                        data[0].T,
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=-lim,
                        vmax=+lim,
                    )
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(images[1], ax=axs[:2], shrink=0.85, label="standardized")
            fig.colorbar(images[2], ax=axs[2], shrink=0.85, label="change")
            text = fig.suptitle("")

            def update(t, images=images, panels=panels, c=c, name=name):
                """Show one frame of the difference."""
                for image, data in zip(images, panels):
                    image.set_data(data[t].T)
                text.set_text(
                    f"{name}: channel {coalign.ANCHOR} − channel {c},"
                    f" sky plane, frame {t}"
                )
                return images

            animation = matplotlib.animation.FuncAnimation(
                fig, update, frames=num_time, interval=250, blit=False
            )
            path = out / f"difference_{name.replace(' ', '')}_ch{c}.gif"
            animation.save(
                path, writer=matplotlib.animation.PillowWriter(fps=FPS), dpi=80
            )
            plt.close(fig)
            print(f"wrote {path}", flush=True)

    fig, axs = plt.subplots(
        1, len(coalign.LINES), figsize=(11.0, 4.2), constrained_layout=True
    )
    for ax, name in zip(np.atleast_1d(axs), coalign.LINES):
        for i, c in enumerate(others):
            rms = summary[(name, c)]
            ax.plot(rms["uncorrected"], "--", color=f"C{i}", label=f"channel {c}")
            ax.plot(rms["corrected"], "-", color=f"C{i}")
        ax.set_title(f"{name}    dashed: uncorrected, solid: coaligned")
        ax.set_xlabel("frame")
    np.atleast_1d(axs)[0].set_ylabel("rms sky-plane difference (standardized)")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize="small")
    path = out / "rms_difference_vs_time.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
