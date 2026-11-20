import argparse

import pytest

from library import sampling


def make_args(**kwargs):
    defaults = dict(sample_at_first=False, sample_every_n_steps=None, sample_every_n_epochs=None, sample_prompts=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("# comment\na cat --w 512 --h 768 --d 1\na dog --n blurry --s 20\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "kwargs, epoch, steps, expected",
    [
        ({}, None, 0, False),
        ({"sample_at_first": True}, 0, 0, True),
        ({"sample_at_first": True}, None, 10, False),  # no schedule
        ({"sample_every_n_steps": 10}, None, 10, True),
        ({"sample_every_n_steps": 10}, None, 11, False),
        ({"sample_every_n_steps": 10}, 1, 10, False),  # the end of an epoch
        ({"sample_every_n_epochs": 2}, 2, 13, True),
        ({"sample_every_n_epochs": 2}, 3, 13, False),
        ({"sample_every_n_epochs": 2}, None, 10, False),
        ({"sample_every_n_epochs": 2, "sample_every_n_steps": 10}, None, 10, False),  # steps are ignored
    ],
)
def test_should_sample_images(kwargs, epoch, steps, expected):
    assert sampling.should_sample_images(make_args(**kwargs), epoch, steps) is expected


def test_forced_prompts_override_the_schedule_and_the_prompt_file(prompt_file):
    args = make_args(sample_prompts=None)
    assert not sampling.has_sample_prompts(args)
    forced = [{"prompt": "forced", "enum": 100}]
    with sampling.force_sample_prompts(forced):
        assert sampling.should_sample_images(args, None, 7)
        assert sampling.has_sample_prompts(args)  # no prompt file is needed
        prompts = sampling.get_sample_prompts(args)
        assert prompts == forced
        prompts[0]["prompt"] = "modified by the sampling function"
        assert forced[0]["prompt"] == "forced"  # a copy is returned
    assert not sampling.should_sample_images(args, None, 7)

    args = make_args(sample_prompts=prompt_file)
    assert sampling.has_sample_prompts(args)
    assert [p["prompt"] for p in sampling.get_sample_prompts(args)] == ["a cat", "a dog"]


def test_collect_saved_samples():
    sampling.notify_sample_saved("before.png")  # nobody collects: ignored
    with sampling.collect_saved_samples() as outer:
        sampling.notify_sample_saved("1.png")
        with sampling.collect_saved_samples() as inner:
            sampling.notify_sample_saved("2.png")
        sampling.notify_sample_saved("3.png")
    sampling.notify_sample_saved("after.png")
    assert outer == ["1.png", "2.png", "3.png"]
    assert inner == ["2.png"]
    assert sampling._saved_sample_collectors == []


def test_build_forced_prompts_from_the_prompt_file(prompt_file):
    args = make_args(sample_prompts=prompt_file)

    prompts = sampling.build_forced_prompts(args)  # all prompts in the file
    assert [p["prompt"] for p in prompts] == ["a cat", "a dog"]
    assert [p["enum"] for p in prompts] == [100, 101]  # not to overwrite the scheduled samples

    prompts = sampling.build_forced_prompts(args, indices=[1], overrides={"seed": 42, "width": None})
    assert len(prompts) == 1
    assert prompts[0]["prompt"] == "a dog"
    assert prompts[0]["negative_prompt"] == "blurry"
    assert prompts[0]["seed"] == 42
    assert "width" not in prompts[0]  # None is ignored

    for bad in [[2], [-1], ["0"], [True]]:
        with pytest.raises(ValueError):
            sampling.build_forced_prompts(args, indices=bad)


def test_verify_forced_prompts_are_cached():
    cached = {"a cat": object(), "": object(), "blurry": object()}
    sampling.verify_forced_prompts_are_cached([{"prompt": "a cat"}, {"prompt": "a cat", "negative_prompt": "blurry"}], cached)
    with pytest.raises(ValueError, match="the prompt `a dog`"):
        sampling.verify_forced_prompts_are_cached([{"prompt": "a cat"}, {"prompt": "a dog"}], cached)
    with pytest.raises(ValueError, match="the negative prompt `ugly`"):
        sampling.verify_forced_prompts_are_cached([{"prompt": "a cat", "negative_prompt": "ugly"}], cached)


def test_build_forced_prompts_from_text(prompt_file):
    args = make_args(sample_prompts=None)
    prompts = sampling.build_forced_prompts(args, prompts=["a bird --w 1024 --h 512 --d 3", {"prompt": "a fish", "scale": 4.0}])
    assert prompts[0] == {"prompt": "a bird", "width": 1024, "height": 512, "seed": 3, "enum": 100}
    assert prompts[1] == {"prompt": "a fish", "scale": 4.0, "enum": 101}

    # the indices come first, then the texts
    args = make_args(sample_prompts=prompt_file)
    prompts = sampling.build_forced_prompts(args, prompts=["a bird"], indices=[0])
    assert [p["prompt"] for p in prompts] == ["a cat", "a bird"]

    args = make_args(sample_prompts=None)
    with pytest.raises(ValueError):
        sampling.build_forced_prompts(args)  # no prompt file and no prompts
    with pytest.raises(ValueError):
        sampling.build_forced_prompts(args, indices=[0])
    with pytest.raises(ValueError):
        sampling.build_forced_prompts(args, prompts=[1])
    with pytest.raises(ValueError):
        sampling.build_forced_prompts(args, prompts=["a bird"], overrides=["seed"])


@pytest.fixture
def prompt_file_with_reserved(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("a cat --s 20\na bird --n blurry --s 0\na dog\n", encoding="utf-8")
    return str(path)


def test_reserved_prompts_are_cached_but_not_scheduled(prompt_file_with_reserved):
    args = make_args(sample_prompts=prompt_file_with_reserved)

    # `--s 0` is kept as 0, and the prompt is loaded (the callers cache the text encoder outputs of the loaded prompts)
    loaded = sampling.load_prompts(prompt_file_with_reserved)
    assert [p["prompt"] for p in loaded] == ["a cat", "a bird", "a dog"]
    assert [sampling.is_reserved_prompt(p) for p in loaded] == [False, True, False]

    # the scheduled sampling skips it. `enum` is the position in the file, so the file names do not change
    scheduled = sampling.get_sample_prompts(args)
    assert [p["prompt"] for p in scheduled] == ["a cat", "a dog"]
    assert [p["enum"] for p in scheduled] == [0, 2]


def test_reserved_prompts_in_forced_sampling(prompt_file_with_reserved):
    args = make_args(sample_prompts=prompt_file_with_reserved)

    # without indices: the same prompts as the scheduled sampling
    assert [p["prompt"] for p in sampling.build_forced_prompts(args)] == ["a cat", "a dog"]

    # by index: generated with the default steps of the model (no `sample_steps`), or with the override
    prompts = sampling.build_forced_prompts(args, indices=[1])
    assert prompts[0]["prompt"] == "a bird" and prompts[0]["negative_prompt"] == "blurry"
    assert "sample_steps" not in prompts[0]
    prompts = sampling.build_forced_prompts(args, indices=[1, 0], overrides={"sample_steps": 12})
    assert [p["sample_steps"] for p in prompts] == [12, 12]

    # 0 steps is never passed to the sampling functions
    prompts = sampling.build_forced_prompts(args, prompts=["a fish --s 0"], indices=[0], overrides={"sample_steps": 0})
    assert all("sample_steps" not in p for p in prompts)


def test_forced_sampling_of_a_prompt_file_with_reserved_prompts_only(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("a bird --s 0\n", encoding="utf-8")
    args = make_args(sample_prompts=str(path))
    assert sampling.get_sample_prompts(args) == []
    with pytest.raises(ValueError, match="by index"):
        sampling.build_forced_prompts(args)
    assert [p["prompt"] for p in sampling.build_forced_prompts(args, indices=[0])] == ["a bird"]


def test_reserved_prompt_in_dict_format(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text('[{"prompt": "a cat"}, {"prompt": "a bird", "sample_steps": 0}]', encoding="utf-8")
    args = make_args(sample_prompts=str(path))
    assert [p["prompt"] for p in sampling.get_sample_prompts(args)] == ["a cat"]
