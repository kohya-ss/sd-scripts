"""``library.sampling.get_my_scheduler`` builds every ``--sample_sampler`` choice with the installed diffusers."""

import importlib.util

import pytest

from library import sampling
from library.sampling import check_sampler_requirements, get_my_scheduler

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# every choice of --sample_sampler (library/args.py) except the scipy-dependent lms / k_lms
SAMPLERS = [
    "ddim",
    "pndm",
    "euler",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
]


@pytest.mark.parametrize("v_parameterization", [False, True])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_every_sampler_builds_and_sets_timesteps(sampler, v_parameterization):
    scheduler = get_my_scheduler(sample_sampler=sampler, v_parameterization=v_parameterization)
    scheduler.set_timesteps(4)
    assert len(scheduler.timesteps) >= 4
    if v_parameterization:
        assert scheduler.config.prediction_type == "v_prediction"


def test_dpmsolver_uses_sigma_min_final_sigmas():
    # diffusers rejects final_sigmas_type="zero" (its default) for the non-++ algorithm
    scheduler = get_my_scheduler(sample_sampler="dpmsolver", v_parameterization=False)
    assert scheduler.config.algorithm_type == "dpmsolver"
    assert scheduler.config.final_sigmas_type == "sigma_min"
    # dpmsolver++ keeps the diffusers default so its output is unchanged
    scheduler = get_my_scheduler(sample_sampler="dpmsolver++", v_parameterization=False)
    assert scheduler.config.final_sigmas_type == "zero"


def test_dpmsingle_has_no_steps_offset():
    scheduler = get_my_scheduler(sample_sampler="dpmsingle", v_parameterization=False)
    assert "steps_offset" not in scheduler.config


@pytest.mark.parametrize("sampler", [s for s in SAMPLERS if s != "dpmsingle"])
def test_other_samplers_keep_steps_offset_one(sampler):
    scheduler = get_my_scheduler(sample_sampler=sampler, v_parameterization=False)
    assert scheduler.config.steps_offset == 1


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy is not installed")
def test_lms_builds_when_scipy_is_available():
    scheduler = get_my_scheduler(sample_sampler="lms", v_parameterization=False)
    scheduler.set_timesteps(4)


@pytest.mark.parametrize("sampler", ["lms", "k_lms"])
def test_lms_without_scipy_raises_a_clear_error(monkeypatch, sampler):
    monkeypatch.setattr(sampling.importlib.util, "find_spec", lambda name: None if name == "scipy" else object())
    with pytest.raises(ImportError, match="pip install scipy"):
        check_sampler_requirements(sampler)
    with pytest.raises(ImportError, match="pip install scipy"):
        get_my_scheduler(sample_sampler=sampler, v_parameterization=False)


def test_check_sampler_requirements_ignores_other_samplers(monkeypatch):
    monkeypatch.setattr(sampling.importlib.util, "find_spec", lambda name: None)
    for sampler in SAMPLERS:
        check_sampler_requirements(sampler)
