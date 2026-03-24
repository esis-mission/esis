Roadmap
=======

- ESIS Distortion Model

  - Pull merit functions and optimization routines out of notebooks and into codebase.
  - Save results of optimization runs into format that can be read into an instrument instance.
  - From distortion fits build ``InterpolatedSystem`` class that fits a desired polynomial
    to raytrace for faster mapping from source to detector.

- CTIS Package

  - Calculate weights from interpolated system.
  - Transpose weights for backward operation.
  - Define standard interface for all inversion methods.
    (forward, backward, data, merit, inner/outer loop filtering?
    to be ready for:

    - CNN
    - MART/RL (other iterative methods)
    - Possibly sklearn linear regression?

  - Figure out the best way to store results in a common object so we can examine
    convergence/progress and compare methods easily.

- Data Synthesis

  - Forward model from VDEM to detector.
  - Generate synthetic ESIS-II images from simulations.
  - Add noise "correctly".

- Papers

  - ESIS-I Instrument Paper (ASAP)
  - CNN Inversion Paper
  - ESIS-I Full Detector Inversion (big event analysis)
  - ESIS-II pre-flight Science?
  - ESIS-II mission/results
