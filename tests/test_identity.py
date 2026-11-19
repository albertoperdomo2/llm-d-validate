import pytest

from llmdv.identity import IdentityError, build, differences, parse, validate_lane


def test_lane_identity_defaults_to_manual_and_accepts_the_ref_alias():
    assert parse(["accelerator=amd", "llm-d-ref=guide-1"]) == {
        "accelerator": "amd",
        "llm_d_ref": "guide-1",
        "trigger": "manual",
    }


def test_unknown_identity_dimensions_are_usage_errors():
    with pytest.raises(IdentityError, match="llmdv_version"):
        parse(["llmdv_version=old"])


def test_profile_dimensions_are_accepted_as_assertions():
    assert parse(["model=model-a", "engine=vllm"]) == {
        "model": "model-a",
        "engine": "vllm",
        "trigger": "manual",
    }


def test_profile_assertion_mismatch_is_a_usage_error():
    with pytest.raises(IdentityError, match="model"):
        validate_lane({"model": "wrong"}, {"model": "model-a"})


def test_identity_derives_engine_and_gateway_from_role_layer_implementations():
    class Layer:
        def __init__(self, role, implementation):
            self.role = role
            self.implementation = implementation

    class Profile:
        guide = "guide-a"
        engine = "legacy-engine"
        gateway = "legacy-gateway"
        model = type("Model", (), {"public_id": "model-a"})()
        ingresses = (Layer("engine", "vllm"), Layer("gateway", "litellm"))

    value = build(Profile(), {}, "digest")
    assert value["engine"] == "vllm"
    assert value["gateway"] == "litellm"


def test_identity_has_fixed_dimensions_and_reports_non_configuration_notes():
    class Profile:
        guide = "guide-a"
        model = type("Model", (), {"public_id": "model-a"})()
        ingresses = ()

    value = build(
        Profile(), {"accelerator": "amd", "provider": "acme", "trigger": "nightly"}, "sha256:p"
    )
    assert set(value) == {
        "guide",
        "accelerator",
        "engine",
        "provider",
        "gateway",
        "model",
        "llm_d_ref",
        "llmdv_version",
        "profile_digest",
        "trigger",
    }
    assert differences(value, {**value, "llm_d_ref": "guide-b"}) == ["llm_d_ref: None -> 'guide-b'"]
