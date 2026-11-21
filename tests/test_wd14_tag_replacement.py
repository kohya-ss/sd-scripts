import pytest

from finetune.tag_images_by_wd14_tagger import process_tag_replacement


def test_replaces_tags_in_place():
    # the CSV (default_format) path ignores the return value and relies on in-place mutation
    general_tags = ["1girl", "solo", "long_hair"]
    process_tag_replacement(general_tags, "solo,alone;long_hair,long hair")
    assert general_tags == ["1girl", "alone", "long hair"]


def test_single_tag_lookup_only_changes_the_source_tag():
    # the tag_mapping.json path replaces one tag at a time and uses the return value
    replacements = "solo,alone;long_hair,long hair"
    assert process_tag_replacement(["solo"], replacements)[0] == "alone"
    assert process_tag_replacement(["long_hair"], replacements)[0] == "long hair"
    assert process_tag_replacement(["blue_eyes"], replacements)[0] == "blue_eyes"


def test_escaped_comma_and_semicolon():
    tags = ["aa,bb", "cc;dd", "ee"]
    process_tag_replacement(tags, "aa\\,bb,AA\\,BB;cc\\;dd,CC\\;DD")
    assert tags == ["AA,BB", "CC;DD", "ee"]


def test_malformed_replacement_raises():
    with pytest.raises(AssertionError):
        process_tag_replacement(["solo"], "solo")
