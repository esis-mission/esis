#!/usr/bin/env python3
"""
Render the baseline and corrected inversions side by side, over the flight.

Three families of output, all with time as the movie axis:

``doppler_<line>.gif``
    The intensity-weighted mean velocity of one line, baseline against
    corrected, with their difference beside them.  The spurious red-to-blue
    ramp the coalignment removes lives in these maps, so the difference
    panel is close to a picture of the correction itself.

``residual_ch<c>.gif``
    The converged residual of one channel in units of its own uncertainty,
    baseline against corrected.  The third panel is the *change in
    magnitude*, so blue is where the corrected reconstruction fits the data
    better and red is where it fits worse.

``chi2_vs_time.png`` / ``ramp_vs_time.png``
    The two scalar summaries against frame number, which say whether the
    improvement holds over the whole flight or only near the frame the
    correction was fitted on.

The drift of the scene across the grid is measured once, on the baseline,
and undone in both arms: co-registering them separately would fold a
different shift into each and make the difference panels meaningless.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib
import sys
import time as _time

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

TOOLING = pathlib.Path.home() / "repos/esis/docs/reports/level4_tempest"


def plane_gradient(
    value: np.ndarray,
    weight: np.ndarray,
    inside: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit an intensity-weighted plane to a map and return its gradient.

    ``value`` is indexed ``(field_x, field_y)``, and the two gradients come
    back in that order — the change across the full width of the field
    along ``field_x`` first, then along ``field_y``, so each is directly the
    size of the red-to-blue ramp along that axis.
    """
    num_x, num_y = value.shape
    index_x, index_y = np.mgrid[0:num_x, 0:num_y]
    coordinate_x = 2 * index_x / (num_x - 1) - 1
    coordinate_y = 2 * index_y / (num_y - 1) - 1

    good = inside & np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if good.sum() < 100:
        return np.nan, np.nan, np.nan

    a = np.stack([np.ones(good.sum()), coordinate_x[good], coordinate_y[good]], axis=-1)
    w = np.sqrt(weight[good])
    solution, *_ = np.linalg.lstsq(a * w[:, None], value[good] * w, rcond=None)
    return solution[0], 2 * solution[1], 2 * solution[2]


def maps(level_4, drift) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the co-registered intensity and mean velocity of every line.

    Returns two arrays shaped ``(line, time, x, y)``: the intensity as it
    comes, the velocity in km/s.
    """
    intensity = level_4.intensity
    velocity = level_4.velocity_mean

    axis_line = level_4.axis_line
    axis_time = level_4.axis_time
    num_time = level_4.shape[axis_time]

    out_i, out_v = [], []
    for i in range(level_4.num_line):
        frames_i = [
            level_4._index_xy(intensity[{axis_line: i, axis_time: t}])
            for t in range(num_time)
        ]
        frames_v = [
            level_4._index_xy(velocity[{axis_line: i, axis_time: t}].to(u.km / u.s))
            for t in range(num_time)
        ]
        out_i.append(np.stack(level_4._coregistered(frames_i, drift)))
        out_v.append(np.stack(level_4._coregistered(frames_v, drift)))

    return np.stack(out_i), np.stack(out_v)


def animate_triptych(
    left: np.ndarray,
    right: np.ndarray,
    difference: np.ndarray,
    titles: tuple[str, str, str],
    limit: float,
    limit_difference: float,
    label: str,
    label_difference: str,
    suptitle: str,
    frames: np.ndarray,
    cmap: str = "RdBu_r",
) -> matplotlib.animation.FuncAnimation:
    """
    Animate three maps of the same field across time.

    Every panel is transposed on display so the detector or field ``x``
    axis runs horizontally, which is the orientation the rest of the
    project's figures use.
    """
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=True)

    norm = matplotlib.colors.Normalize(-limit, limit)
    norm_difference = matplotlib.colors.Normalize(-limit_difference, limit_difference)

    images = []
    for ax, data, title, n in zip(
        axs,
        (left, right, difference),
        titles,
        (norm, norm, norm_difference),
    ):
        image = ax.imshow(data[0].T, origin="lower", cmap=cmap, norm=n)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(image)

    fig.colorbar(images[1], ax=axs[:2], label=label, shrink=0.85)
    fig.colorbar(images[2], ax=axs[2], label=label_difference, shrink=0.85)
    text = fig.suptitle(f"{suptitle} — frame {frames[0]}")

    def update(t: int):
        for image, data in zip(images, (left, right, difference)):
            image.set_data(data[t].T)
        text.set_text(f"{suptitle} — frame {frames[t]}")
        return (*images, text)

    return matplotlib.animation.FuncAnimation(
        fig, update, frames=len(frames), interval=200, blit=False
    )


def main() -> None:
    """Render every comparison product."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--residual-baseline")
    parser.add_argument("--residual-corrected")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pitch", type=float, default=0.75)
    parser.add_argument("--num-velocity", type=int, default=24)
    parser.add_argument(
        "--limit-velocity",
        type=float,
        default=20.0,
        help=(
            "half-range of the Doppler colour scale in km/s; tighter than the"
            " usual 100 because the point here is the field-wide ramp, which"
            " is a few km/s and invisible on a scale set by the line core"
        ),
    )
    parser.add_argument("--limit-difference", type=float, default=8.0)
    parser.add_argument(
        "--percentile-intensity",
        type=float,
        default=20.0,
        help="blank the velocity below this percentile of the line intensity",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--no-coregister", action="store_true")
    parser.add_argument(
        "--tooling",
        default=str(TOOLING),
        help="directory holding render_movies.py and tied_config.py",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.tooling)
    import render_movies

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = _time.perf_counter()
    arms = {
        name: render_movies.build(path, args.pitch, args.num_velocity)
        for name, path in (
            ("baseline", args.baseline),
            ("corrected", args.corrected),
        )
    }
    print(f"products rebuilt: {_time.perf_counter() - t0:.0f} s", flush=True)

    baseline = arms["baseline"]
    frames = np.asarray(np.load(args.baseline)["frames"])

    drift = None
    if not args.no_coregister:
        drift = arms["baseline"].drift()
        offset = drift.to_value(u.arcsec)
        print(
            f"drift measured on the baseline and applied to both arms:"
            f" x {offset[:, 0].min():+.2f} to {offset[:, 0].max():+.2f},"
            f" y {offset[:, 1].min():+.2f} to {offset[:, 1].max():+.2f} arcsec",
            flush=True,
        )

    labels = [baseline.label(i) for i in range(baseline.num_line)]
    data = {name: maps(arm, drift) for name, arm in arms.items()}
    print(f"maps extracted: {_time.perf_counter() - t0:.0f} s", flush=True)

    # --- Doppler movies, one per line ---------------------------------------
    for i, label in enumerate(labels):
        intensity = data["baseline"][0][i]
        floor = np.nanpercentile(intensity, args.percentile_intensity)
        faint = intensity < floor

        panels = {}
        for name in ("baseline", "corrected"):
            velocity = data[name][1][i].copy()
            velocity[faint] = np.nan
            panels[name] = velocity

        animation = animate_triptych(
            panels["baseline"],
            panels["corrected"],
            panels["corrected"] - panels["baseline"],
            ("baseline", "coaligned", "coaligned − baseline"),
            args.limit_velocity,
            args.limit_difference,
            "mean velocity (km/s)",
            "change (km/s)",
            f"{label} Doppler",
            frames,
        )
        name = label.replace(" ", "_")
        animation.save(out / f"doppler_{name}.gif", writer="pillow", fps=args.fps)
        plt.close(animation._fig)
        print(f"saved doppler_{name}.gif", flush=True)

    # --- residual movies, one per channel -----------------------------------
    if args.residual_baseline and args.residual_corrected:
        residual = {
            "baseline": np.load(args.residual_baseline)["residual"],
            "corrected": np.load(args.residual_corrected)["residual"],
        }
        num_channel = residual["baseline"].shape[1]
        for c in range(num_channel):
            b = residual["baseline"][:, c]
            k = residual["corrected"][:, c]
            limit = float(np.nanpercentile(np.abs(b), 99))
            animation = animate_triptych(
                b,
                k,
                np.abs(k) - np.abs(b),
                ("baseline", "coaligned", "|coaligned| − |baseline|"),
                limit,
                limit / 2,
                "residual (σ)",
                "change in |residual| (σ)",
                f"channel {c} residual",
                residual_frames(args.residual_baseline),
            )
            animation.save(out / f"residual_ch{c}.gif", writer="pillow", fps=args.fps)
            plt.close(animation._fig)
            print(f"saved residual_ch{c}.gif", flush=True)

    # --- scalar summaries ----------------------------------------------------
    chi2 = {
        name: np.asarray(arm.mean_chi_squared.ndarray) for name, arm in arms.items()
    }
    num_channel = chi2["baseline"].shape[1]

    fig, axs = plt.subplots(
        1, num_channel, figsize=(3.2 * num_channel, 3.4), constrained_layout=True
    )
    for c, ax in enumerate(np.atleast_1d(axs)):
        ax.plot(frames, chi2["baseline"][:, c], "-", color="0.4", label="baseline")
        ax.plot(frames, chi2["corrected"][:, c], "-", color="C3", label="coaligned")
        ax.set_title(f"channel {c}")
        ax.set_xlabel("frame")
        ax.set_yscale("log")
    np.atleast_1d(axs)[0].set_ylabel("mean $\\chi^2$")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize="small")
    fig.savefig(out / "chi2_vs_time.png", dpi=140)
    plt.close(fig)
    print("saved chi2_vs_time.png", flush=True)

    # every cell is offered to the fit; `plane_gradient` drops the ones
    # outside the support on its own, since they carry no intensity
    inside = np.ones(data["baseline"][1][0][0].shape, dtype=bool)
    gradients = {
        name: np.array(
            [
                [
                    plane_gradient(data[name][1][i][t], data[name][0][i][t], inside)
                    for t in range(len(frames))
                ]
                for i in range(len(labels))
            ]
        )
        for name in arms
    }

    fig, (ax, ax_component) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), constrained_layout=True
    )
    for i, label in enumerate(labels):
        for name, style in (("baseline", "--"), ("corrected", "-")):
            ramp = np.hypot(gradients[name][i, :, 1], gradients[name][i, :, 2])
            ax.plot(
                frames,
                ramp,
                style,
                color=f"C{i}",
                label=label if style == "-" else None,
            )
    ax.set_xlabel("frame")
    ax.set_ylabel("velocity ramp across the field (km/s)")
    ax.set_title("dashed: baseline    solid: coaligned")
    ax.legend(frameon=False, fontsize="small", ncol=2)

    # the components carry more than the magnitude does: the correction does
    # not act equally on the two field axes, and which axis is left with a
    # ramp is the clue to what the remaining error is
    index_strongest = int(
        np.nanargmax(
            np.hypot(
                gradients["baseline"][:, :, 1], gradients["baseline"][:, :, 2]
            ).mean(axis=1)
        )
    )
    for component, axis in ((1, "field x"), (2, "field y")):
        for name, style in (("baseline", "--"), ("corrected", "-")):
            ax_component.plot(
                frames,
                gradients[name][index_strongest, :, component],
                style,
                color="C0" if component == 1 else "C3",
                label=f"{axis} {name}",
            )
    ax_component.axhline(0, color="0.7", lw=0.8)
    ax_component.set_xlabel("frame")
    ax_component.set_ylabel("ramp component (km/s)")
    ax_component.set_title(f"{labels[index_strongest]}: ramp by axis")
    ax_component.legend(frameon=False, fontsize="small", ncol=2)

    fig.savefig(out / "ramp_vs_time.png", dpi=140)
    plt.close(fig)
    print("saved ramp_vs_time.png", flush=True)

    print(f"TOTAL: {_time.perf_counter() - t0:.0f} s", flush=True)


def residual_frames(path: str) -> np.ndarray:
    """Read the frame numbers recorded beside a saved residual cube."""
    return np.asarray(np.load(path)["frames"])


if __name__ == "__main__":
    main()
