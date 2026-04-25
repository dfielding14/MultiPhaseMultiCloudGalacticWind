"""Pytest configuration for deterministic local JAX execution."""

import os


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
