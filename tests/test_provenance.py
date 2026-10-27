"""The transform spec, the provenance hash and the derived output keys (``vestigo.provenance``).

The bytes are wire-frozen: the canonical JSON and the derived keys are pinned here as
literals, so a change to the encoding fails before it changes a stored key.
"""

from __future__ import annotations

import hashlib

import pytest

from vestigo.provenance import (
    TransformSpec,
    checkpoint_output_key,
    compute_provenance_hash,
    derive_subkey,
    parse_transform_spec,
    run_output_key,
)
from vestigo.store import ArtifactStore, ArtifactVersion


class TestTransformSpec:
    def test_canonical_json_is_the_pinned_encoding(self):
        spec = TransformSpec(
            executor="x@v1",
            params={"b": [1, 2.5, True, None, {"n": "\u00e9"}], "a": "s"},
            inputs=["t://b", "t://a"],
        )
        assert spec.canonical_json == (
            '{"executor": "x@v1", "inputs": ["t://b", "t://a"], '
            '"params": {"a": "s", "b": [1, 2.5, true, null, {"n": "\\u00e9"}]}}'
        )
        assert parse_transform_spec(spec.canonical_json) == spec

    def test_inputs_keep_their_order_and_input_hashes_do_not(self):
        ab = TransformSpec("x@v1", {}, ["a", "b"])
        ba = TransformSpec("x@v1", {}, ["b", "a"])
        assert compute_provenance_hash(["h1", "h2"], ab) != compute_provenance_hash(
            ["h1", "h2"], ba
        )
        assert compute_provenance_hash(["h1", "h2"], ab) == compute_provenance_hash(
            ["h2", "h1"], ab
        )

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({1: "a"}, "non-string key 1"),
            ({"a": {"b": {2: "c"}}}, "non-string key 2"),
            ({"a": (1, 2)}, "params\\['a'\\] is a tuple"),
            ({"a": [1, (2,)]}, "params\\['a'\\]\\[1\\] is a tuple"),
            ({"a": float("nan")}, "which JSON cannot carry"),
            ({"a": [float("inf")]}, "which JSON cannot carry"),
            ({"a": {1, 2}}, "is a set"),
            ({"a": b"bytes"}, "is a bytes"),
        ],
        ids=["int_key", "nested_int_key", "tuple", "nested_tuple", "nan", "inf", "set", "bytes"],
    )
    def test_params_json_cannot_carry_are_refused_at_construction(self, params, message):
        with pytest.raises(ValueError, match=message):
            TransformSpec("x@v1", params, [])

    def test_params_must_be_a_dict(self):
        with pytest.raises(ValueError, match="not a dict"):
            TransformSpec("x@v1", [("a", 1)], [])  # type: ignore[arg-type]

    def test_a_spec_is_not_hashable(self):
        with pytest.raises(TypeError):
            hash(TransformSpec("x@v1", {}, []))


# json.loads reads NaN and Infinity, which TransformSpec refuses: a writer from before
# that check, or a foreign store, could have stored this text.
NAN_SPEC = '{"executor": "x@v1", "inputs": [], "params": {"content_type": "arrow/ipc", "w": NaN}}'


class TestParseTransformSpec:
    @pytest.mark.parametrize(
        "text",
        [None, "", "not json", "[1, 2]", NAN_SPEC, '{"params": {"a": [Infinity]}}'],
        ids=["none", "empty", "not_json", "not_an_object", "nan", "infinity"],
    )
    def test_a_spec_that_cannot_be_rebuilt_reads_as_unrecorded(self, text):
        assert parse_transform_spec(text) is None

    def test_a_missing_or_mistyped_field_reads_as_empty(self):
        assert parse_transform_spec('{"params": [1], "inputs": "a"}') == TransformSpec("", {}, [])

    def test_a_stored_row_whose_spec_cannot_be_rebuilt_reads_as_unrecorded(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        record = ArtifactVersion(
            id="foreign",
            version=1,
            state="ready",
            provenance_hash="P-foreign",
            schema_json="{}",
            row_count=1,
            byte_size=5,
            created_at=1.0,
            transform_spec=NAN_SPEC,
        )
        store.import_artifact(record, b"bytes")

        row = store.get_artifact("foreign", 1)
        assert row is not None and row.transform_spec == NAN_SPEC
        assert (row.spec, row.params) == (None, {})
        assert store.versions_declaring("arrow/ipc") == []
        store.close()


class TestDerivedKeys:
    PARENT = hashlib.sha256(b"parent").hexdigest()

    def test_the_wire_format_is_parent_and_labels_joined_by_colons(self):
        key = derive_subkey(self.PARENT, "build=b1", "release=3", "model")
        assert key == hashlib.sha256(f"{self.PARENT}:build=b1:release=3:model".encode()).hexdigest()
        assert checkpoint_output_key(self.PARENT, "b1", 3, "model") == key
        assert run_output_key(self.PARENT, "b1", "model") == derive_subkey(
            self.PARENT, "build=b1", "model"
        )

    def test_a_label_holding_the_separator_is_refused(self):
        with pytest.raises(ValueError, match="contains ':'"):
            derive_subkey(self.PARENT, "a:b")
        with pytest.raises(ValueError, match="contains ':'"):
            derive_subkey(self.PARENT, "a", "b:c")
        assert derive_subkey(self.PARENT, "a", "b") == (
            "045421da72d44725f6c296c07a327dffe2857c0b6b7d7a44d69f01447d126b44"
        )
