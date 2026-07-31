#!/usr/bin/env python3
"""
Compare the parallel Level-4 inversion against the alternatives.

Run on tempest (inside the esis environment, with ``ESIS_CACHE_DIR`` set as
in ``submit_level_4.py``) after the corresponding jobs have finished::

    python compare.py chain
    python compare.py weights --frames 0,7,15,22,29

``chain`` compares the embarrassingly-parallel inversion (every frame from
the same Gaussian seed) against the sequential warm-start chain, frame by
frame.  ``weights`` compares frames inverted with the shared reference-frame
weights against the same frames inverted with weights rebuilt from the
fitted per-frame pointing.
"""

import argparse

import numpy as np


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Measure the (x, y) shift of `b` relative to `a` by cross-correlation.

    Parameters
    ----------
    a
        The reference image.
    b
        The shifted image.
    """
    f = np.fft.rfft2(np.nan_to_num(a))
    g = np.fft.rfft2(np.nan_to_num(b))
    correlation = np.fft.irfft2(f * g.conj(), s=a.shape)
    index = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = [i if i <= n // 2 else i - n for i, n in zip(index, correlation.shape)]
    return -shift[0], -shift[1]


def _summarize(name: str, a, b, axis_time: str) -> None:
    """
    Print the frame-by-frame comparison of two Level-4 products.

    Parameters
    ----------
    name
        The label of the comparison.
    a
        The reference product (or single-frame product).
    b
        The product compared against it.
    axis_time
        The name of the time axis.
    """
    intensity_a = a.intensity
    intensity_b = b.intensity
    index_line = a.num_line - 1

    num_time = a.shape[axis_time]
    print(f"=== {name} ===")
    for t in range(num_time):
        chi2_a = np.asarray(a.mean_chi_squared[{axis_time: t}].ndarray)
        chi2_b = np.asarray(b.mean_chi_squared[{axis_time: t}].ndarray)

        map_a = intensity_a[{axis_time: t, a.axis_line: index_line}]
        map_b = intensity_b[{axis_time: t, b.axis_line: index_line}]
        xy_a = np.asarray(map_a.ndarray.value)
        xy_b = np.asarray(map_b.ndarray.value)

        shift = _shift(xy_a, xy_b)
        difference = np.nanmean(np.abs(xy_b - xy_a)) / np.nanmean(np.abs(xy_a))

        print(
            f"frame {t:3d}:"
            f" chi2 a {np.array2string(chi2_a, precision=3)}"
            f" b {np.array2string(chi2_b, precision=3)}"
            f" | shift (cells) {shift}"
            f" | mean |diff|/|a| {difference:.4f}",
            flush=True,
        )


def main() -> None:
    """Parse the command line and run the requested comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["chain", "weights"])
    parser.add_argument("--frames", default="0,7,15,22,29")
    args = parser.parse_args()

    import esis

    data = esis.flights.f1.data

    if args.mode == "chain":
        parallel = data.level_4_parallel()
        chain = data.level_4()
        _summarize(
            name="warm-start chain (a) vs parallel seed (b)",
            a=chain,
            b=parallel,
            axis_time=chain.axis_time,
        )

    elif args.mode == "weights":
        for index in [int(i) for i in args.frames.split(",")]:
            shared = data.level_4_frame(index, weights_shared=True)
            perframe = data.level_4_frame(index, weights_shared=False)
            _summarize(
                name=f"frame {index}: shared (a) vs per-frame (b) weights",
                a=shared,
                b=perframe,
                axis_time=shared.axis_time,
            )


if __name__ == "__main__":
    main()
