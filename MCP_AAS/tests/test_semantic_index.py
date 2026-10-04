"""Retrieval: flattening, descriptors, concept enrichment, ranking, caching.

No network and no embedding calls. Vectors are supplied directly so the ranking
and context-expansion behaviour can be asserted exactly rather than
approximately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from mcp_aas.semantic.index import (
    Index,
    IndexError_,
    Node,
    _slug,
    concept_map,
    enrich,
    search,
    walk,
)


def prop(id_short: str, value: Any = None, model_type: str = "Property") -> Dict[str, Any]:
    return {"modelType": model_type, "idShort": id_short, "value": value}


def submodel(*elements: Dict[str, Any], sm_id: str = "urn:sm", id_short: str = "SM") -> Dict[str, Any]:
    return {
        "modelType": "Submodel",
        "id": sm_id,
        "idShort": id_short,
        "submodelElements": list(elements),
    }


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------

def test_nested_collections_flatten_to_dotted_paths() -> None:
    sm = submodel(
        {
            "modelType": "SubmodelElementCollection",
            "idShort": "CapacityEnergyVoltage",
            "value": [prop("RatedCapacity"), prop("MinVoltage")],
        }
    )
    assert [n.path for n in walk(sm)] == [
        "CapacityEnergyVoltage",
        "CapacityEnergyVoltage.RatedCapacity",
        "CapacityEnergyVoltage.MinVoltage",
    ]


def test_list_items_are_addressed_by_position() -> None:
    """The source composed `parent.idShort`, but AASd-120 forbids a list child
    an idShort -- so every item became `Markings.None` and overwrote the last."""
    sm = submodel(
        {
            "modelType": "SubmodelElementList",
            "idShort": "Markings",
            "typeValueListElement": "SubmodelElementCollection",
            "value": [
                {"modelType": "SubmodelElementCollection", "value": [prop("MarkingName", "CE")]},
                {"modelType": "SubmodelElementCollection", "value": [prop("MarkingName", "UKCA")]},
            ],
        }
    )
    paths = [n.path for n in walk(sm)]
    assert paths == [
        "Markings",
        "Markings[0]",
        "Markings[0].MarkingName",
        "Markings[1]",
        "Markings[1].MarkingName",
    ]
    assert len(set(paths)) == len(paths)


def test_parents_are_recorded_for_boundary_expansion() -> None:
    sm = submodel(
        {
            "modelType": "SubmodelElementCollection",
            "idShort": "Address",
            "value": [prop("Street")],
        }
    )
    nodes = {n.path: n for n in walk(sm)}
    assert nodes["Address"].parent is None
    assert nodes["Address.Street"].parent == "Address"


def test_element_values_are_rendered_per_type() -> None:
    sm = submodel(
        prop("Serial", "SN-1"),
        {"modelType": "MultiLanguageProperty", "idShort": "Name", "value": [{"en": "Cell"}]},
        {"modelType": "Range", "idShort": "Temp", "min": "-20", "max": "60"},
        {"modelType": "SubmodelElementCollection", "idShort": "Group", "value": []},
    )
    values = {n.path: n.value for n in walk(sm)}
    assert values["Serial"] == "SN-1"
    assert values["Name"] == "en: Cell"
    assert values["Temp"] == "min=-20, max=60"
    assert values["Group"] is None


def test_elements_without_an_id_short_outside_a_list_are_skipped() -> None:
    """Outside a list an idShort is mandatory; a nameless element is unaddressable."""
    sm = submodel({"modelType": "Property", "value": "orphan"}, prop("Serial"))
    assert [n.path for n in walk(sm)] == ["Serial"]


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------

def test_id_short_is_split_into_words() -> None:
    node = Node(path="RatedCapacity", id_short="RatedCapacity", model_type="Property", parent=None)
    assert "title: Rated Capacity" in node.descriptor()


def test_acronyms_and_underscores_split_too() -> None:
    node = Node(
        path="TemperatureRangeIdleState_LowerBoundary",
        id_short="TemperatureRangeIdleState_LowerBoundary",
        model_type="Property",
        parent=None,
    )
    assert "Temperature Range Idle State Lower Boundary" in node.descriptor()


def test_the_parent_collection_is_part_of_the_meaning() -> None:
    """`MinVoltage` inside `CapacityEnergyVoltage` means more than `MinVoltage`."""
    node = Node(
        path="TechnicalPropertyAreas.CapacityEnergyVoltage.MinVoltage",
        id_short="MinVoltage",
        model_type="Property",
        parent="TechnicalPropertyAreas.CapacityEnergyVoltage",
    )
    assert "within: Capacity Energy Voltage" in node.descriptor()


def test_citation_boilerplate_is_stripped_by_shape() -> None:
    node = Node(
        path="RatedCapacity",
        id_short="RatedCapacity",
        model_type="Property",
        parent=None,
        definition="rated capacity\n\nDIN DKE Spec 99100 chapter reference: 6.7.2.2",
    )
    text = node.descriptor()
    assert "rated capacity" in text
    assert "DIN DKE" not in text and "6.7.2.2" not in text


@pytest.mark.parametrize(
    "definition",
    [
        "the mass\n\nIEC 63278 section reference: 4.2",
        "the mass\nISO 1234 clause reference: 9",
    ],
)
def test_other_standards_are_stripped_too(definition: str) -> None:
    """The pattern matches the shape of a citation, not one standard's name."""
    node = Node(path="M", id_short="M", model_type="Property", parent=None, definition=definition)
    assert node.descriptor().rstrip(", ").endswith("the mass")


def test_a_real_definition_mentioning_a_standard_survives() -> None:
    node = Node(
        path="M",
        id_short="M",
        model_type="Property",
        parent=None,
        definition="conformance with DIN DKE Spec 99100 as assessed by the manufacturer",
    )
    assert "DIN DKE Spec 99100" in node.descriptor()


def test_values_are_excluded_by_default_and_opt_in() -> None:
    """Search answers 'where does this belong'; a slot's meaning is not its value."""
    node = Node(path="Temp", id_short="Temp", model_type="Property", parent=None, value="36.8")
    assert "36.8" not in node.descriptor()
    assert "value: 36.8" in node.descriptor(with_values=True)


def test_list_items_always_carry_their_value() -> None:
    """Three MarkingName properties have identical schemas; only the value differs."""
    node = Node(
        path="Markings[1].MarkingName",
        id_short="MarkingName",
        model_type="Property",
        parent="Markings[1]",
        value="UKCA conformity marking",
    )
    assert node.in_list
    assert "value: UKCA conformity marking" in node.descriptor()


def test_a_long_value_cannot_swamp_the_schema() -> None:
    node = Node(
        path="L[0]", id_short="Note", model_type="Property", parent="L", value="x" * 500
    )
    assert len(node.descriptor()) < 300


# ---------------------------------------------------------------------------
# concept descriptions
# ---------------------------------------------------------------------------

def _concept(identifier: str, definition: str, name: str = "", unit: str = "") -> Dict[str, Any]:
    return {
        "modelType": "ConceptDescription",
        "id": identifier,
        "embeddedDataSpecifications": [
            {
                "dataSpecificationContent": {
                    "modelType": "DataSpecificationIec61360",
                    "definition": [{"language": "en", "text": definition}],
                    "preferredName": [{"language": "en", "text": name}] if name else [],
                    "unit": unit,
                }
            }
        ],
    }


def test_concept_ids_are_stripped_before_matching() -> None:
    """The reference repository stores several with a leading space."""
    concepts = concept_map([_concept(" 0173-1#02-AAO220#003", "a remark")])
    assert "0173-1#02-AAO220#003" in concepts


def test_enrichment_attaches_definition_name_and_unit() -> None:
    node = Node(
        path="RatedCapacity",
        id_short="RatedCapacity",
        model_type="Property",
        parent=None,
        semantic_id="0173-1#02-ABL869#002",
    )
    concepts = concept_map(
        [_concept("0173-1#02-ABL869#002", "rated capacity", name="rated capacity", unit="Ah")]
    )
    assert enrich([node], concepts) == 1
    assert node.unit == "Ah"
    assert "unit: Ah" in node.descriptor()


def test_a_semantic_id_with_no_concept_is_left_alone() -> None:
    """19 ECLASS ids in the live repository have no ConceptDescription at all."""
    node = Node(
        path="X", id_short="X", model_type="Property", parent=None,
        semantic_id="0173-1#02-ABL588#001", description="fallback text",
    )
    assert enrich([node], {}) == 0
    assert "fallback text" in node.descriptor()


def test_a_concept_without_iec61360_content_still_indexes() -> None:
    concepts = concept_map([{"modelType": "ConceptDescription", "id": "urn:x"}])
    assert concepts["urn:x"] == {"definition": "", "preferred_name": "", "unit": ""}


# ---------------------------------------------------------------------------
# ranking and boundary expansion
# ---------------------------------------------------------------------------

def _index_of(paths: List[str], parents: List[Any], vectors: List[List[float]]) -> Index:
    return Index(
        submodel_id="urn:sm",
        id_short="SM",
        model="test",
        nodes=[
            Node(path=p, id_short=p.rsplit(".", 1)[-1], model_type="Property", parent=parent)
            for p, parent in zip(paths, parents)
        ],
        vectors=np.asarray(vectors, dtype="float32"),
    )


def test_ranking_is_by_cosine_similarity() -> None:
    index = _index_of(
        ["A.x", "A.y", "A.z"], ["A", "A", "A"], [[1, 0], [0, 1], [0.7, 0.7]]
    )
    hits = search(index, np.asarray([1.0, 0.0]), top_k=3)
    assert [h.node.path for h in hits] == ["A.x", "A.z", "A.y"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_returned_so_a_threshold_is_possible() -> None:
    """The source reported ranks with no similarity; `aas match` needs numbers."""
    index = _index_of(["A.x", "A.y"], ["A", "A"], [[1, 0], [0, 1]])
    hits = search(index, np.asarray([1.0, 0.0]), top_k=2, min_score=0.5)
    assert [h.node.path for h in hits] == ["A.x"]


def test_a_hit_comes_back_inside_its_boundary() -> None:
    """A bare top-k list invites pairing a real hit with an imagined sibling."""
    index = _index_of(
        ["A", "A.x", "A.y", "B.z"],
        [None, "A", "A", "B"],
        [[0, 0], [1, 0], [0, 1], [0.5, 0.5]],
    )
    hit = search(index, np.asarray([1.0, 0.0]), top_k=1)[0]
    assert hit.node.path == "A.x"
    assert hit.parent is not None and hit.parent.path == "A"
    assert hit.siblings == ["y"]      # not z: it lives under a different parent


def test_top_k_larger_than_the_corpus_is_not_an_error() -> None:
    index = _index_of(["A.x"], ["A"], [[1, 0]])
    assert len(search(index, np.asarray([1.0, 0.0]), top_k=50)) == 1


def test_an_empty_index_returns_nothing_rather_than_raising() -> None:
    empty = Index(submodel_id="urn:sm", id_short="SM", model="test")
    assert search(empty, np.asarray([1.0, 0.0])) == []


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def test_an_index_round_trips_through_disk(tmp_path: Path) -> None:
    index = _index_of(["A.x", "A.y"], ["A", "A"], [[1, 0], [0, 1]])
    index.enriched = 2
    index.with_values = True
    index.save(tmp_path)

    loaded = Index.load("urn:sm", tmp_path)
    assert [n.path for n in loaded.nodes] == ["A.x", "A.y"]
    assert loaded.enriched == 2
    assert loaded.with_values is True
    np.testing.assert_allclose(loaded.vectors, index.vectors)


def test_loading_an_unbuilt_index_names_the_command_that_builds_it(tmp_path: Path) -> None:
    with pytest.raises(IndexError_, match="aas index urn:missing"):
        Index.load("urn:missing", tmp_path)


def test_cached_listing_reports_the_mode(tmp_path: Path) -> None:
    index = _index_of(["A.x"], ["A"], [[1, 0]])
    index.save(tmp_path)
    (row,) = Index.cached(tmp_path)
    assert row["submodel_id"] == "urn:sm"
    assert row["idShort"] == "SM"
    assert row["nodes"] == 1
    assert row["with_values"] is False
    assert row["model"] == "test"
    # An index is a snapshot; the listing has to say how old it is.
    assert row["built_at"] > 0


def test_dropping_an_index_removes_both_files(tmp_path: Path) -> None:
    """A submodel can be deleted while its index survives, and a search would
    then return paths that no longer exist."""
    _index_of(["A.x"], ["A"], [[1, 0]]).save(tmp_path)
    assert Index.drop("urn:sm", tmp_path) is True
    assert Index.cached(tmp_path) == []
    assert Index.drop("urn:sm", tmp_path) is False


def test_ids_differing_only_in_a_path_segment_get_different_files() -> None:
    """Submodel ids are URIs; a slug built from readable characters alone collides."""
    a = _slug("https://example.com/ids/sm/TechnicalData/1/0")
    b = _slug("https://example.com/ids/sm/TechnicalData/1/1")
    assert a != b
    assert "/" not in a and ":" not in a


def test_the_slug_stays_readable() -> None:
    assert "TechnicalData" in _slug("https://example.com/ids/sm/TechnicalData").replace("-", "")


# ---------------------------------------------------------------------------
# structural filters and deduplication
# ---------------------------------------------------------------------------

def _node(path, **kw):
    from mcp_aas.semantic.index import Node

    base = dict(id_short=path.rsplit(".", 1)[-1], model_type="Property", parent=None)
    base.update(kw)
    return Node(path=path, **base)


def _index(*nodes):
    import numpy as np

    from mcp_aas.semantic.index import Index

    return Index(
        submodel_id="*", id_short="test", model="fake",
        nodes=list(nodes), vectors=np.eye(len(nodes), dtype="float32"),
    )


def test_a_filter_narrows_the_rows_before_ranking() -> None:
    """The point of filtering is that ranking never sees the excluded rows.

    Discarding after the fact would still make the right answer compete with
    every irrelevant element for a top-k slot, which is the contest it loses.
    """
    from mcp_aas.semantic.index import Filter, rows_matching

    index = _index(
        _node("A", model_type="Property"),
        _node("B", model_type="SubmodelElementCollection"),
        _node("C", model_type="Property", unit="kWh"),
    )
    assert rows_matching(index, Filter(model_type="Property")) == [0, 2]
    assert rows_matching(index, Filter(unit="kwh")) == [2]
    assert rows_matching(index, None) == [0, 1, 2]


def test_filter_fields_are_a_conjunction() -> None:
    from mcp_aas.semantic.index import Filter, rows_matching

    index = _index(
        _node("A", model_type="Property", unit="kWh"),
        _node("B", model_type="Property", unit="V"),
    )
    assert rows_matching(index, Filter(model_type="Property", unit="V")) == [1]
    assert rows_matching(index, Filter(model_type="Range", unit="V")) == []


def test_an_empty_filter_constrains_nothing() -> None:
    from mcp_aas.semantic.index import Filter

    assert Filter().empty()
    assert not Filter(unit="V").empty()


def test_a_filter_that_matches_nothing_returns_no_hits_rather_than_all() -> None:
    """Failing open here would silently undo the caller's constraint."""
    import numpy as np

    from mcp_aas.semantic.index import Filter, search

    index = _index(_node("A"), _node("B"))
    hits = search(index, np.array([1.0, 0.0], dtype="float32"),
                  where=Filter(model_type="Blob"))
    assert hits == []


def test_duplicate_elements_collapse_into_one_hit() -> None:
    """Four of five near-misses on the repository-wide set were rank 4.

    The first three slots held one wrong element repeated across three
    duplicate submodels. Returning k identical rows spends a step budget
    saying one thing, so copies collapse and the other locations are reported
    on the surviving hit.
    """
    import numpy as np

    from mcp_aas.semantic.index import search

    index = _index(
        _node("Same", semantic_id="urn:c:1", submodel="urn:sm:a"),
        _node("Same", semantic_id="urn:c:1", submodel="urn:sm:b"),
        _node("Same", semantic_id="urn:c:1", submodel="urn:sm:c"),
        _node("Other", semantic_id="urn:c:2", submodel="urn:sm:a"),
    )
    query = np.array([1.0, 0.9, 0.8, 0.7], dtype="float32")

    hits = search(index, query, top_k=2)
    assert [h.node.path for h in hits] == ["Same", "Other"]
    assert sorted(hits[0].also_in) == ["urn:sm:b", "urn:sm:c"]


def test_deduplication_can_be_turned_off() -> None:
    import numpy as np

    from mcp_aas.semantic.index import search

    index = _index(
        _node("Same", semantic_id="urn:c:1", submodel="urn:sm:a"),
        _node("Same", semantic_id="urn:c:1", submodel="urn:sm:b"),
    )
    hits = search(index, np.array([1.0, 0.9], dtype="float32"), top_k=2, dedupe=False)
    assert len(hits) == 2


def test_elements_sharing_a_path_but_not_a_type_stay_separate() -> None:
    """Two different kinds of element at one path are not the same fact."""
    import numpy as np

    from mcp_aas.semantic.index import search

    index = _index(
        _node("Name", model_type="Property", submodel="urn:sm:a"),
        _node("Name", model_type="MultiLanguageProperty", submodel="urn:sm:b"),
    )
    hits = search(index, np.array([1.0, 0.9], dtype="float32"), top_k=2)
    assert len(hits) == 2


def test_the_dedupe_key_does_not_depend_on_a_semantic_id() -> None:
    """31% of the reference repository carries none.

    A key including `semanticId` never matches for those elements, so they
    silently stop collapsing -- duplicates return for exactly the elements with
    the least other information to identify them. The ablation found no
    measured difference between the candidate keys, so the tie breaks on what
    every element is guaranteed to have.
    """
    import numpy as np

    from mcp_aas.semantic.index import search

    index = _index(
        _node("Same", semantic_id=None, submodel="urn:sm:a"),
        _node("Same", semantic_id=None, submodel="urn:sm:b"),
    )
    hits = search(index, np.array([1.0, 0.9], dtype="float32"), top_k=2)
    assert len(hits) == 1, "elements without a semanticId failed to collapse"
    assert hits[0].also_in == ["urn:sm:b"]


def test_siblings_do_not_cross_submodels() -> None:
    """A boundary assembled from two submodels is an invented boundary.

    `GeneralInformation` exists in dozens of submodels here; without the owner
    check the reported neighbourhood would mix elements that never sit beside
    each other, which is exactly the invention the boundary exists to prevent.
    """
    import numpy as np

    from mcp_aas.semantic.index import search

    index = _index(
        _node("Box.A", parent="Box", submodel="urn:sm:a"),
        _node("Box.B", parent="Box", submodel="urn:sm:a"),
        _node("Box.C", parent="Box", submodel="urn:sm:b"),
    )
    hits = search(index, np.array([1.0, 0.0, 0.0], dtype="float32"), top_k=1)
    assert hits[0].siblings == ["B"]
