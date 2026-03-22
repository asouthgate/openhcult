import numpy as np

from hcultinf.inference import fit_parametric_monotonic_spline


def test_parametric_spline_fits_logistic():
    # True relationship: x(z) is logistic, so z(x) goes near-vertical at the extremes.
    # This is the hard case that motivates arc-length reparameterisation.
    z_true = np.linspace(0.05, 0.95, 60)
    x_true = 1.0 / (1.0 + np.exp(-12.0 * (z_true - 0.5)))

    x_mid = 0.5 * (x_true[:-1] + x_true[1:])
    dz_dx = np.diff(z_true) / np.diff(x_true)

    anchor_idx = np.linspace(0, len(x_true) - 1, 8).astype(int)
    x_anchors = x_true[anchor_idx]
    z_anchors = z_true[anchor_idx]

    sx, sz = fit_parametric_monotonic_spline(
        x_anchors, z_anchors, x_mid, dz_dx, knots=6, k=3, w_der=1.0
    )

    s_fine = np.linspace(0, 1, 1000)
    x_fit = sx(s_fine)
    z_fit = sz(s_fine)

    import matplotlib.pyplot as plt

    plt.plot(x_true, z_true)
    plt.plot(x_fit, z_fit)
    plt.show()

    # z must be monotonically decreasing along the curve
    assert np.all(np.diff(z_fit) <= 0.01)

    # Fitted curve must pass close to each interior anchor in (x, z) space
    for xa, za in zip(x_anchors[1:-1], z_anchors[1:-1]):
        dist = np.sqrt((x_fit - xa) ** 2 + (z_fit - za) ** 2)
        assert dist.min() < 0.05, f"Curve too far from anchor ({xa:.3f}, {za:.3f})"
