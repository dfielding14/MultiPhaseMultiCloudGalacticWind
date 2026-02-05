#!/usr/bin/env python
"""
Example: diagnose integration outcomes and event triggers.

Runs one viable and one intentionally non-viable model, then prints
termination diagnostics from `Solution.sol`.
"""

import warnings

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.constants import kpc


def event_names() -> list[str]:
    """Event order for default supersonic starts (Mach > 1)."""
    return [
        "subsonic_transition",
        "wind_negative",
        "cold_wind_transition",
        "all_clouds_frozen",
        "cloud_density_low",
        "cloud_velocity_low",
        "negative_pressure",
        "negative_density",
        "nan_state",
        "step_size_floor",
    ]


def summarize(label: str, solution) -> None:
    print(f"\n{label}")
    print(f"  status: {solution.sol.status}")
    print(f"  message: {solution.sol.message}")
    print(f"  final radius: {solution.r[-1]:.2f} kpc")

    names = event_names()
    if hasattr(solution.sol, "t_events"):
        triggered = False
        for i, t_hits in enumerate(solution.sol.t_events):
            if len(t_hits) > 0:
                triggered = True
                name = names[i] if i < len(names) else f"event_{i}"
                print(f"  triggered: {name} at r={t_hits[-1] / kpc:.2f} kpc")
        if not triggered:
            print("  triggered: none (reached integration bound)")


def main() -> None:
    viable = WindModel(
        SFR=10.0,
        eta_M=0.3,
        eta_M_cold=0.1,
        eta_E=1.0,
        v_circ=150.0,
        r_max_kpc=20.0,
        N_cloud_species=6,
        rtol=1e-6,
        atol=1e-8,
    ).run()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nonviable = WindModel(
            SFR=10.0,
            eta_M=0.1,
            eta_M_cold=1.0,
            eta_E=0.1,
            v_circ=150.0,
            r_max_kpc=20.0,
            N_cloud_species=6,
            rtol=1e-6,
            atol=1e-8,
        ).run()

    summarize("Viable run", viable)
    summarize("Intentionally non-viable run", nonviable)


if __name__ == "__main__":
    main()
