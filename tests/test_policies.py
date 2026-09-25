import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langgraph_instrumentation import (
    CaptureMode,
    CapturePolicy,
    MessageProjector,
    MetadataProjector,
    RedactionPolicy,
    Sanitizer,
    TokenUsage,
    Trace,
    TraceId,
    UsageExtractor,
)


def test_sanitizer_handles_cycles_shared_values_and_depth() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    shared = {"value": 1}
    sanitizer = Sanitizer(max_depth=2)

    assert sanitizer.sanitize(cyclic) == ["<cycle>"]
    assert sanitizer.sanitize([shared, shared]) == [
        {"value": 1},
        {"value": 1},
    ]
    assert sanitizer.sanitize({"one": {"two": {"three": 3}}}) == {"one": {"two": "<max-depth>"}}


def test_sanitizer_bounds_strings_collections_and_bytes() -> None:
    sanitizer = Sanitizer(
        max_string_length=4,
        max_collection_items=2,
        max_byte_length=3,
    )

    assert sanitizer.sanitize("abcdef") == "abcd<truncated:2>"
    assert sanitizer.sanitize([1, 2, 3]) == [1, 2, "<truncated:1>"]
    assert sanitizer.sanitize({"a": 1, "b": 2, "c": 3}) == {
        "a": 1,
        "b": 2,
        "<truncated>": 1,
    }
    assert sanitizer.sanitize(b"secret") == {
        "type": "bytes",
        "length": 6,
        "base64": "c2Vj",
        "truncated": 3,
    }
    assert CapturePolicy.full().capture(b"secret") == {
        "type": "bytes",
        "length": 6,
        "base64": "c2VjcmV0",
    }


def test_bounded_sanitizer_does_not_materialize_entire_mapping() -> None:
    class CountingMapping(Mapping[str, int]):
        def __init__(self) -> None:
            self.visited = 0

        def __len__(self) -> int:
            return 1_000_000

        def __iter__(self) -> Iterator[str]:
            for index in range(1_000_000):
                self.visited += 1
                if self.visited > 4:
                    raise AssertionError("sanitizer consumed beyond its bounded lookahead")
                yield str(index)

        def __getitem__(self, key: str) -> int:
            return int(key)

    value = CountingMapping()

    assert Sanitizer(max_collection_items=2).sanitize(value) == {
        "0": 0,
        "1": 1,
        "<truncated>": 999_998,
    }
    assert value.visited == 3


def test_sanitizer_never_calls_user_string_or_repr_for_keys_and_sets() -> None:
    class Dangerous:
        def __hash__(self) -> int:
            return 1

        def __str__(self) -> str:
            raise AssertionError("must not call user __str__")

        def __repr__(self) -> str:
            raise AssertionError("must not call user __repr__")

    dangerous = Dangerous()
    sanitizer = Sanitizer(max_collection_items=5)

    assert sanitizer.sanitize({dangerous: 1}) == {"<key:Dangerous>": 1}
    assert sanitizer.sanitize({dangerous}) == ["<unsupported:Dangerous>"]


def test_redaction_is_case_insensitive_and_supports_custom_predicate() -> None:
    policy = RedactionPolicy(
        keys=frozenset({"password"}),
        predicate=lambda path, key, value: key.endswith("_private") or "private" in path,
    )
    sanitizer = Sanitizer(redaction=policy)

    assert sanitizer.sanitize(
        {
            "PASSWORD": "secret",
            "profile_private": "hidden",
            "nested": {"value": "visible"},
        }
    ) == {
        "PASSWORD": "<redacted>",
        "profile_private": "<redacted>",
        "nested": {"value": "visible"},
    }


def test_redaction_uses_original_key_before_key_truncation() -> None:
    assert Sanitizer(max_string_length=4).sanitize({"password": "must-not-leak"}) == {
        "pass<truncated:4>": "<redacted>"
    }


def test_redaction_paths_preserve_original_parent_keys() -> None:
    policy = RedactionPolicy(
        keys=frozenset(),
        predicate=lambda path, key, value: "private" in path,
    )

    assert Sanitizer(redaction=policy, max_string_length=4).sanitize(
        {"private": {"value": "must-not-leak"}}
    ) == {"priv<truncated:3>": {"valu<truncated:1>": "<redacted>"}}


def test_unsupported_objects_use_a_safe_type_marker() -> None:
    @dataclass
    class Dangerous:
        secret: str

        def __str__(self) -> str:
            return self.secret

    sanitized = Sanitizer().sanitize(Dangerous("must-not-leak"))

    assert sanitized == "<unsupported:Dangerous>"
    assert "must-not-leak" not in json.dumps(sanitized)


def test_capture_modes_control_payload_retention() -> None:
    payload = {"prompt": "top secret", "items": [1, 2, 3]}

    assert CapturePolicy.none().capture(payload) is None
    metadata = CapturePolicy.metadata().capture(payload)
    assert metadata == {"type": "mapping", "size": 2}
    assert "top secret" not in json.dumps(metadata)
    assert CapturePolicy.bounded(max_collection_items=1).capture(payload) == {
        "prompt": "top secret",
        "<truncated>": 1,
    }
    assert CapturePolicy.full().capture(payload) == payload


def test_sanitization_does_not_mutate_input_and_conforms_to_trace_attributes() -> None:
    source = {"items": [1, 2], "token": "secret"}
    sanitized = Sanitizer().sanitize(source)

    assert source == {"items": [1, 2], "token": "secret"}
    assert sanitized == {"items": [1, 2], "token": "<redacted>"}
    assert isinstance(sanitized, dict)
    trace = Trace(
        trace_id=TraceId(1),
        name="safe",
        service_name="test",
        start_time_unix_ns=1,
        attributes=sanitized,
    )
    assert trace.to_dict()["attributes"] == sanitized


def test_message_metadata_capture_does_not_retain_message_bodies() -> None:
    projector = MessageProjector(capture=CapturePolicy.metadata())
    message = HumanMessage(
        id="message-1",
        content="private prompt",
        additional_kwargs={"password": "private password"},
    )

    projected = projector.project(message)
    serialized = json.dumps(projected)

    assert projected["content"] == {"type": "str", "length": 14}
    assert "private prompt" not in serialized
    assert "private password" not in serialized


def test_message_none_capture_omits_all_message_payload_fields() -> None:
    projected = MessageProjector(capture=CapturePolicy.none()).project(
        HumanMessage(id="private-id", name="private-name", content="private-body")
    )

    assert projected == {"type": "human"}


def test_message_projector_replaces_repeated_bodies_with_references() -> None:
    projector = MessageProjector(capture=CapturePolicy.bounded())
    message = HumanMessage(id="message-1", content="hello")

    first = projector.project(message)
    repeated = projector.project(message)

    assert first["content"] == "hello"
    assert repeated == {"type": "human", "id": "message-1", "repeated": True}


def test_message_projector_does_not_dedupe_changed_content_with_same_id() -> None:
    projector = MessageProjector(capture=CapturePolicy.bounded())

    projector.project(HumanMessage(id="message-1", content="first"))
    changed = projector.project(HumanMessage(id="message-1", content="second"))

    assert changed["content"] == "second"


def test_message_dedupe_cache_uses_fixed_size_hashes() -> None:
    projector = MessageProjector(capture=CapturePolicy.bounded(max_string_length=10))

    projector.project(HumanMessage(id="x" * 100_000, content="hello"))

    assert all(len(key) == 32 for key in projector._seen_message_ids)


def test_message_projector_bounds_seen_id_cache() -> None:
    projector = MessageProjector(
        capture=CapturePolicy.bounded(),
        max_seen_message_ids=1,
    )

    projector.project(HumanMessage(id="one", content="first"))
    projector.project(HumanMessage(id="two", content="second"))

    assert projector.project(HumanMessage(id="one", content="first"))["content"] == "first"


def test_metadata_projector_has_configurable_selection_and_redaction() -> None:
    metadata = {
        "thread_id": "thread-1",
        "custom": {"value": 1},
        "authorization": "bearer secret",
        "ignored": "value",
    }
    projector = MetadataProjector(
        include_keys=None,
        exclude_keys=frozenset({"ignored"}),
    )

    assert projector.project(metadata) == {
        "authorization": "<redacted>",
        "custom": {"value": 1},
        "thread_id": "thread-1",
    }
    assert MetadataProjector().project(metadata) == {"thread_id": "thread-1"}


def test_metadata_projector_validates_keys_before_sorting() -> None:
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        MetadataProjector(include_keys=None).project(cast(Mapping[str, object], {"valid": 1, 2: 2}))


def test_metadata_projection_is_bounded() -> None:
    metadata = {f"key-{index}": index for index in range(100)}
    projected = MetadataProjector(
        include_keys=None,
        capture=CapturePolicy.bounded(max_collection_items=2),
    ).project(metadata)

    assert projected == {"key-0": 0, "key-1": 1, "<truncated>": 98}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
            TokenUsage(2, 3, 5),
        ),
        (
            {"usage": {"input_tokens": "4", "output_tokens": "6"}},
            TokenUsage(4, 6, 10),
        ),
        (
            {"usage_metadata": {"input_token_count": 7, "output_token_count": 8}},
            TokenUsage(7, 8, 15),
        ),
    ],
)
def test_usage_extractor_normalizes_llm_output_shapes(payload, expected: TokenUsage) -> None:
    response = LLMResult(generations=[], llm_output=payload)

    assert UsageExtractor().extract(response) == expected


def test_usage_extractor_reads_message_metadata_and_tolerates_missing_data() -> None:
    message = AIMessage(
        content="response",
        usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    response = LLMResult(generations=[[ChatGeneration(message=message)]])

    assert UsageExtractor().extract(response) == TokenUsage(2, 3, 5)
    assert UsageExtractor().extract(object()) == TokenUsage()


def test_usage_extractor_skips_malformed_candidates_and_supports_provider_aliases() -> None:
    @dataclass
    class Response:
        llm_output: dict[str, object]
        usage_metadata: dict[str, object]

    response = Response(
        llm_output={"token_usage": {"prompt_tokens": -1, "completion_tokens": True}},
        usage_metadata={
            "prompt_token_count": 4,
            "candidates_token_count": 6,
            "total_token_count": 10,
        },
    )

    assert UsageExtractor().extract(response) == TokenUsage(4, 6, 10)


def test_usage_extractor_ignores_pathologically_large_decimal_strings() -> None:
    @dataclass
    class Response:
        llm_output: dict[str, object]
        usage_metadata: dict[str, object]

    response = Response(
        llm_output={"token_usage": {"total_tokens": "9" * 10_000}},
        usage_metadata={"input_tokens": 1, "output_tokens": 2},
    )

    assert UsageExtractor().extract(response) == TokenUsage(1, 2, 3)


def test_invalid_policy_configuration_fails_clearly() -> None:
    with pytest.raises(TypeError, match="CaptureMode"):
        CapturePolicy(mode=cast(CaptureMode, CaptureMode.METADATA.value))
    with pytest.raises(ValueError, match="greater than zero"):
        CapturePolicy(max_depth=0)
    with pytest.raises(ValueError, match="must not be empty"):
        RedactionPolicy(keys=frozenset({""}))
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(input_tokens=-1)


def test_capture_mode_values_are_stable() -> None:
    assert [mode.value for mode in CaptureMode] == ["none", "metadata", "bounded", "full"]
