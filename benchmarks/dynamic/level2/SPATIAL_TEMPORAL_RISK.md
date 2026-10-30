# Spatial-temporal voltage risk metrics

The current transient-stability KPI averages normalized voltage excursions
across buses. That system-level scalar can hide a severe excursion at one bus
or a brief extreme event. This extension reports spatial and temporal
diagnostics while leaving the existing severity score unchanged.

Under-voltage is normalized as `clip((0.95 - V)/(0.95 - 0.90), 0, 1)`;
over-voltage as `clip((V - 1.05)/(1.10 - 1.05), 0, 1)`. Metrics include
trapezoidally integrated mean and worst-bus exposures, the fraction of buses
that ever violate a limit, an upper-tail mean of sample-level spatial-average
violations, and the longest contiguous interval where both interval endpoints
violate a threshold. Nonuniform time sampling is supported. Duplicate or
decreasing timestamps and non-finite values are rejected.

The upper-tail mean is CVaR-like in spirit, but is not formal CVaR over a
probability distribution. These descriptive diagnostics do not replace
protection limits, stability criteria, or the generator-speed instability flag.

## Offline validation
Run `python -m unittest -v test_spatial_temporal_risk.py` from this directory.
Tests use synthetic trajectories and need neither PowerFactory nor network access.
