"""Entity matching: bands, fallbacks, unit checks, value typing, plan round-trip.

No network and no model. The arbitrator is a callable, so the part that has to
be right -- which band an entity lands in, what happens when arbitration
declines, what gets warned about -- is asserted exactly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pytest

from mcp_aas.semantic.match import (
    THETA_HIGH,
    THETA_LOW,
    Candidate,
    Decision,
    ElementSpec,
    Entity,
    MatchError,
    Plan,
    band_for,
    decide,
    element_for_create,
    infer_element_type,
    infer_value_type,
    load_entities,
    partition,
    safe_id_short,
    unit_warning,
    value_for_write,
)


def entity(name: str = "Rated capacity", **kw: Any) -> Entity:
    return Entity(
        entity=name,
        value=kw.get("value", "68"),
        description=kw.get("description", "nominal capacity"),
        unit=kw.get("unit", ""),
        source=kw.get("source", "datasheet.pdf p.3"),
    )


def candidate(path: str, score: float, model_type: str = "Property", **kw: Any) -> Candidate:
    return Candidate(
        path=path,
        id_short=path.rsplit(".", 1)[-1],
        model_type=model_type,
        score=score,
        parent=kw.get("parent", path.rsplit(".", 1)[0] if "." in path else None),
        unit=kw.get("unit", ""),
        definition=kw.get("definition", ""),
        type_value_list_element=kw.get("holds"),
    )


def picks(index: int, reason: str = "because"):
    def arbitrate(_e: Entity, cands: Sequence[Candidate]):
        return (cands[index] if 0 <= index < len(cands) else None), reason

    return arbitrate


def declines(reason: str = "nothing fits"):
    def arbitrate(_e: Entity, _c: Sequence[Candidate]):
        return None, reason

    return arbitrate


# ---------------------------------------------------------------------------
# the three bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [(0.95, "auto"), (0.8, "auto"), (0.79, "arbitrated"), (0.5, "arbitrated"), (0.49, "below")],
)
def test_band_boundaries_are_inclusive_at_the_top(score: float, expected: str) -> None:
    """theta_high is 'at or above'; theta_low is 'below'. Both ends matter."""
    assert band_for(score, THETA_HIGH, THETA_LOW) == expected


def test_a_high_similarity_match_is_taken_without_asking() -> None:
    decision = decide(entity(), [candidate("A.RatedCapacity", 0.91)], arbitrate=picks(0))
    assert decision.action == "write"
    assert decision.band == "auto"
    assert decision.path == "A.RatedCapacity"
    assert "0.9100" in decision.reason


def test_the_middle_band_is_handed_over_and_the_answer_is_recorded() -> None:
    decision = decide(
        entity(),
        [candidate("A.Wrong", 0.66), candidate("A.RatedCapacity", 0.61)],
        arbitrate=picks(1, "capacity, not the wrong one"),
    )
    assert decision.action == "write"
    assert decision.band == "arbitrated"
    assert decision.path == "A.RatedCapacity"
    assert decision.reason == "capacity, not the wrong one"


def test_a_low_similarity_match_is_never_written() -> None:
    """Below theta_low no candidate is worth arbitrating over, let alone taking."""
    decision = decide(entity(), [candidate("A.Unrelated", 0.31)], arbitrate=picks(0))
    assert decision.action == "skip"
    assert decision.band == "below"
    assert decision.path is None


def test_thresholds_are_parameters_not_constants() -> None:
    """They are reported in the paper; a caller must be able to move them."""
    cands = [candidate("A.X", 0.62)]
    assert decide(entity(), cands, theta_high=0.6).band == "auto"
    assert decide(entity(), cands, theta_low=0.7).band == "below"


# ---------------------------------------------------------------------------
# when arbitration is unavailable or declines
# ---------------------------------------------------------------------------

def test_without_an_arbitrator_the_middle_band_stays_unresolved() -> None:
    """A coin flip dressed as a decision is worse than an unanswered entity."""
    decision = decide(entity(), [candidate("A.X", 0.65)])
    assert decision.action == "skip"
    assert decision.band == "arbitrated"
    assert "no arbitration model" in decision.reason


def test_a_declined_arbitration_falls_through_to_hosting() -> None:
    decision = decide(
        entity(),
        [candidate("A.X", 0.65), candidate("Technical", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=picks(0, "belongs among technical properties"),
    )
    assert decision.action == "create"
    assert decision.band == "hosted"
    assert decision.path == "Technical"
    assert decision.reason == "belongs among technical properties"


def test_a_declined_host_leaves_the_entity_unplaced() -> None:
    """Creating an element in the wrong collection is worse than reporting no home."""
    decision = decide(
        entity(),
        [candidate("A.X", 0.65), candidate("Technical", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=declines(),
    )
    assert decision.action == "skip"


def test_create_in_is_the_fallback_when_no_host_is_chosen() -> None:
    """The source hardcoded `TechnicalPropertyAreas`; here the caller names it."""
    decision = decide(
        entity(), [candidate("A.X", 0.2)], create_in="TechnicalPropertyAreas"
    )
    assert decision.action == "create"
    assert decision.path == "TechnicalPropertyAreas"
    assert decision.id_short == "Rated_capacity"


# ---------------------------------------------------------------------------
# candidate partitioning
# ---------------------------------------------------------------------------

def test_shells_and_submodels_are_never_candidates() -> None:
    leaves, containers = partition(
        [
            candidate("Root", 0.9, "Submodel"),
            candidate("Shell", 0.9, "AssetAdministrationShell"),
            candidate("A.X", 0.7),
        ]
    )
    assert [c.path for c in leaves] == ["A.X"]
    assert containers == []


def test_collections_are_hosts_not_write_targets() -> None:
    leaves, containers = partition(
        [candidate("Group", 0.9, "SubmodelElementCollection"), candidate("Group.X", 0.7)]
    )
    assert [c.path for c in leaves] == ["Group.X"]
    assert [c.path for c in containers] == ["Group"]


def test_a_collection_scoring_highest_does_not_become_the_write_target() -> None:
    """Retrieval often ranks the container above its child; writing to a
    collection is not a thing."""
    decision = decide(
        entity(),
        [candidate("Group", 0.95, "SubmodelElementCollection"), candidate("Group.X", 0.85)],
    )
    assert decision.action == "write"
    assert decision.path == "Group.X"


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_a_unit_mismatch_is_warned_about_not_silently_written() -> None:
    """8 years written into a property declared in months is wrong in a way
    nothing downstream catches. The source ignored units entirely."""
    decision = decide(
        entity("Warranty period", unit="a"),
        [candidate("G.WarrantyPeriod", 0.88, unit="month")],
    )
    assert decision.action == "write"
    assert any("unit mismatch" in w for w in decision.warnings)


@pytest.mark.parametrize("left,right", [("Ah", "Ah"), ("A h", "Ah"), ("kg", "KG"), ("", "kg")])
def test_agreeing_or_absent_units_produce_no_warning(left: str, right: str) -> None:
    """An absent unit is unknown, not zero; warning about silence buries the
    warnings that matter."""
    assert unit_warning(left, right) is None


def test_a_value_without_provenance_is_flagged() -> None:
    decision = decide(entity(source=""), [candidate("A.X", 0.9)])
    assert any("no source" in w for w in decision.warnings)


def test_an_entity_with_no_value_is_flagged() -> None:
    decision = decide(entity(value=None), [candidate("A.X", 0.9)])
    assert any("no value" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# value typing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("68", "xs:integer"),
        ("355.2", "xs:double"),
        ("-3", "xs:integer"),
        ("true", "xs:boolean"),
        ("2026-03-14", "xs:date"),
        ("2026-03-14T10:00:00", "xs:dateTime"),
        ("SN-4711", "xs:string"),
        ("", "xs:string"),
        (None, "xs:string"),
    ],
)
def test_value_types_are_inferred(value: Optional[str], expected: str) -> None:
    assert infer_value_type(value) == expected


def test_a_leading_zero_forces_a_string() -> None:
    """`007` as an integer is `7`, and the difference is not recoverable."""
    assert infer_value_type("007") == "xs:string"


def test_a_long_unitless_digit_run_is_an_identifier() -> None:
    """A GTIN is not a quantity. The unit is the signal: measured values carry
    one, identifiers do not."""
    assert infer_value_type("4012345678901") == "xs:string"
    assert infer_value_type("4012345678901", unit="pcs") == "xs:integer"


def test_ordinary_integers_are_untouched_by_the_identifier_rule() -> None:
    assert infer_value_type("2000") == "xs:integer"
    assert infer_value_type("123456789") == "xs:integer"


# ---------------------------------------------------------------------------
# idShorts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Rated capacity", "Rated_capacity"),
        ("Zellspannung (Ü)", "Zellspannung_U"),
        ("2nd voltage", "Entity_2nd_voltage"),
        ("...", "Entity"),
    ],
)
def test_id_shorts_are_made_aasd_002_legal(raw: str, expected: str) -> None:
    assert safe_id_short(raw) == expected


# ---------------------------------------------------------------------------
# entities and plans
# ---------------------------------------------------------------------------

def test_entities_accept_either_key_for_the_name() -> None:
    loaded = load_entities([{"entity": "A", "value": "1"}, {"name": "B", "value": "2"}])
    assert [e.entity for e in loaded] == ["A", "B"]


def test_an_entities_object_with_a_list_inside_is_accepted() -> None:
    assert len(load_entities({"entities": [{"entity": "A"}]})) == 1


def test_a_nameless_entity_is_refused_rather_than_matched_blindly() -> None:
    with pytest.raises(MatchError, match="no name"):
        load_entities([{"value": "68"}])


def test_the_unit_is_not_part_of_the_query() -> None:
    """Embedding it pulls every dimensionless entity toward every unit-bearing
    element; it is a correctness constraint, checked afterwards."""
    assert "Ah" not in entity(unit="Ah").query()


def test_a_plan_round_trips() -> None:
    plan = Plan(submodel_id="urn:sm", theta_high=0.8, theta_low=0.5, model="gpt-x")
    plan.decisions.append(decide(entity(), [candidate("A.X", 0.9)]))
    restored = Plan.from_json(plan.to_json())
    assert restored.submodel_id == "urn:sm"
    assert restored.theta_high == 0.8
    assert [d.path for d in restored.decisions] == ["A.X"]


def test_a_plan_records_the_thresholds_it_was_made_with() -> None:
    """Applying a plan later must not depend on today's defaults."""
    plan = Plan(submodel_id="urn:sm", theta_high=0.72, theta_low=0.4, model=None)
    assert Plan.from_json(plan.to_json()).theta_high == 0.72


def test_something_that_is_not_a_plan_is_refused() -> None:
    with pytest.raises(MatchError, match="not a match plan"):
        Plan.from_json({"hello": "world"})


def test_the_margin_between_first_and_second_is_recorded() -> None:
    decision = decide(entity(), [candidate("A.X", 0.75), candidate("A.Y", 0.55)])
    assert decision.margin == pytest.approx(0.20, abs=1e-4)


def test_a_dead_auto_band_is_reported() -> None:
    """theta_high is calibrated against one descriptor and one embedding model;
    if nothing clears it, every entity goes to arbitration unannounced."""
    plan = Plan(submodel_id="urn:sm", theta_high=0.8, theta_low=0.5, model="gpt-x")
    plan.decisions.append(decide(entity(), [candidate("A.X", 0.75)], arbitrate=picks(0)))
    note = plan.calibration_note()
    assert note is not None and "0.750" in note


def test_no_note_when_the_auto_band_did_fire() -> None:
    plan = Plan(submodel_id="urn:sm", theta_high=0.8, theta_low=0.5, model=None)
    plan.decisions.append(decide(entity(), [candidate("A.X", 0.91)]))
    assert plan.calibration_note() is None


def test_counts_separate_the_bands_from_the_actions() -> None:
    plan = Plan(submodel_id="urn:sm", theta_high=0.8, theta_low=0.5, model="gpt-x")
    plan.decisions.append(decide(entity(), [candidate("A.X", 0.91)]))
    plan.decisions.append(decide(entity(), [candidate("A.Y", 0.6)], arbitrate=picks(0)))
    plan.decisions.append(decide(entity(), [candidate("A.Z", 0.1)]))
    counts = plan.counts()
    assert counts["write"] == 2 and counts["skip"] == 1
    assert counts["auto"] == 1 and counts["arbitrated"] == 1


# ---------------------------------------------------------------------------
# unit spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "document,element",
    [
        ("kg", "kilogram"),
        ("%", "percent"),
        ("Ah", "ampereHour"),
        ("V", "volt"),
        ("min", "minuteUnitOfTime"),
        ("a", "year"),
        ("degC", "\u00b0C"),
        ("\u00baC", "\u00b0C"),          # masculine ordinal vs degree sign
        ("kWh", "kilowattHour"),
    ],
)
def test_the_same_unit_spelled_two_ways_does_not_warn(document: str, element: str) -> None:
    """The repository mixes IEC 61360 long form in ConceptDescriptions with
    symbols in documents. A false warning costs a correct write under
    `aas populate --skip-warnings`."""
    assert unit_warning(document, element) is None


@pytest.mark.parametrize(
    "document,element",
    [("kWh", "Wh"), ("years", "month"), ("mm", "cm"), ("g", "kg")],
)
def test_genuinely_different_units_still_warn(document: str, element: str) -> None:
    """The table normalises spellings; it does not convert magnitudes, which is
    exactly the case that must not be silently accepted."""
    assert unit_warning(document, element) is not None


# ---------------------------------------------------------------------------
# what to create, not just where
# ---------------------------------------------------------------------------

def types(index: int, spec: Optional[ElementSpec], reason: str = "belongs here"):
    """A host callable that answers the type as well as the place."""

    def host(_e: Entity, cands: Sequence[Candidate]):
        return (cands[index] if 0 <= index < len(cands) else None), reason, spec

    return host


@pytest.mark.parametrize(
    "value,unit,name,expected",
    [
        ("250-550", "mm", "Stroke", "Range"),
        ("-20...+60", "°C", "Operating temperature", "Range"),
        ("250 - 550 mm", "", "Stroke", "Range"),
        ("10..20", "", "Pressure range", "Range"),
        ("Motor@en|Motor@de", "", "Designation", "MultiLanguageProperty"),
        ("https://example.com/ds.pdf", "", "Datasheet", "File"),
        ("./docs/manual.pdf", "", "Manual", "File"),
        ("68", "kWh", "Rated capacity", "Property"),
        ("hello", "", "Note", "Property"),
    ],
)
def test_the_value_shape_suggests_the_element_type(
    value: str, unit: str, name: str, expected: str
) -> None:
    assert infer_element_type(entity(name, value=value, unit=unit)).model_type == expected


@pytest.mark.parametrize("value", ["1234-5678", "2026-08-29", "A-1", "007-2"])
def test_two_numbers_around_a_dash_are_not_automatically_a_range(value: str) -> None:
    """An order number and a date both look like an interval and are not.

    The rule demands a second signal -- a unit, or a word saying so -- because
    a Property holding "1234-5678" is readable and a Range built from an order
    number is a defect in the model of a machine.
    """
    assert infer_element_type(
        entity("Order number", value=value, unit="", description="")
    ).model_type == "Property"


def test_a_range_carries_both_bounds_and_the_wider_of_the_two_types() -> None:
    spec = infer_element_type(entity("Stroke", value="250-550.5", unit="mm"))
    assert (spec.model_type, spec.min_value, spec.max_value) == ("Range", "250", "550.5")
    assert spec.value_type == "xs:double"


def test_a_reference_is_never_inferred_only_chosen() -> None:
    """An IRI is indistinguishable from an identifier stored as a string."""
    spec = infer_element_type(entity("Concept", value="0173-1#02-AAO677#002", unit=""))
    assert spec.model_type == "Property"


def test_the_host_model_types_the_element_it_places() -> None:
    decision = decide(
        entity("Operating temperature", value="-20 to 60", unit="°C"),
        [candidate("Technical", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=types(0, ElementSpec(model_type="Range", min_value="-20", max_value="60")),
    )
    assert decision.action == "create"
    assert (decision.model_type, decision.min_value, decision.max_value) == ("Range", "-20", "60")


def test_a_host_that_answers_only_where_still_gets_the_inferred_type() -> None:
    """The shape inference seeds the answer, so a terse model is not a downgrade."""
    decision = decide(
        entity("Designation", value="Motor@en|Motor@de", unit=""),
        [candidate("Nameplate", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=picks(0),
    )
    assert decision.model_type == "MultiLanguageProperty"


def test_the_model_cannot_create_a_type_this_tool_will_not_build() -> None:
    """A collection is a plausible-sounding container for anything, and
    creating one from a single fact loses the fact."""
    decision = decide(
        entity(),
        [candidate("Technical", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=types(0, ElementSpec(model_type="Capability")),
    )
    assert decision.model_type == "Property"
    assert any("created as" in w for w in decision.warnings)


def test_a_range_without_bounds_degrades_to_a_property_and_says_so() -> None:
    decision = decide(
        entity("Note", value="ask the supplier", unit=""),
        [candidate("Technical", 0.6, "SubmodelElementCollection")],
        arbitrate=declines(),
        host=types(0, ElementSpec(model_type="Range")),
    )
    assert decision.model_type == "Property"
    assert any("no min and max" in w for w in decision.warnings)


def test_a_list_dictates_its_children_over_any_judgement() -> None:
    """AASd-108: every child of a SubmodelElementList has the declared type."""
    decision = decide(
        entity("Marking", value="CE", unit=""),
        [candidate("Markings", 0.6, "SubmodelElementList", holds="MultiLanguageProperty")],
        arbitrate=declines(),
        host=types(0, ElementSpec(model_type="Property")),
    )
    assert decision.model_type == "MultiLanguageProperty"
    assert any("SubmodelElementList of MultiLanguageProperty" in w for w in decision.warnings)


def test_a_child_of_a_list_carries_no_id_short() -> None:
    """AASd-120: a list child is addressed by position, not by name."""
    decision = decide(
        entity(),
        [candidate("Markings", 0.6, "SubmodelElementList", holds="Property")],
        arbitrate=declines(),
        host=picks(0),
    )
    assert decision.id_short is None


def test_a_list_of_something_unbuildable_is_reported_not_attempted() -> None:
    decision = decide(
        entity(),
        [candidate("Events", 0.6, "SubmodelElementList", holds="BasicEventElement")],
        arbitrate=declines(),
        host=picks(0),
    )
    assert any("no extracted value can be created as" in w for w in decision.warnings)


def test_writing_into_an_existing_range_records_the_bounds_in_the_plan() -> None:
    """The judgement belongs in the plan a person reviews, not in populate."""
    decision = decide(
        entity("Operating temperature", value="-20...60", unit="°C"),
        [candidate("Technical.Temp", 0.9, "Range", unit="°C")],
    )
    assert decision.action == "write"
    assert (decision.min_value, decision.max_value) == ("-20", "60")


def test_a_range_target_with_an_unreadable_value_is_warned_about() -> None:
    decision = decide(
        entity("Operating temperature", value="ambient", unit=""),
        [candidate("Technical.Temp", 0.9, "Range")],
    )
    assert any("is a Range" in w and "no min and max" in w for w in decision.warnings)


# ---------------------------------------------------------------------------
# applying what the plan says
# ---------------------------------------------------------------------------

def test_a_language_string_is_written_in_the_value_only_shape() -> None:
    """`[{"en": ...}]` into a `$value`; the element shape draws HTTP 500."""
    decision = Decision(
        entity="Designation", value="Motor@en|Motor@de", unit="", source="",
        action="write", band="auto", reason="", path="X",
        model_type="MultiLanguageProperty",
    )
    assert value_for_write(decision) == [{"en": "Motor"}, {"de": "Motor"}]


def test_a_range_is_written_as_an_object_not_a_scalar() -> None:
    decision = Decision(
        entity="Temp", value="-20...60", unit="", source="", action="write",
        band="auto", reason="", path="X", model_type="Range",
        min_value="-20", max_value="60",
    )
    assert value_for_write(decision) == {"min": "-20", "max": "60"}


def test_a_range_write_with_no_bounds_is_refused_before_it_is_sent() -> None:
    decision = Decision(
        entity="Temp", value="ambient", unit="", source="", action="write",
        band="auto", reason="", path="X", model_type="Range",
    )
    with pytest.raises(MatchError):
        value_for_write(decision)


def test_a_file_is_written_with_its_content_type() -> None:
    decision = Decision(
        entity="Datasheet", value="ds.pdf", unit="", source="", action="write",
        band="auto", reason="", path="X", model_type="File",
        content_type="application/pdf",
    )
    assert value_for_write(decision) == {"contentType": "application/pdf", "value": "ds.pdf"}


def test_a_plain_property_is_still_written_as_a_scalar() -> None:
    decision = Decision(
        entity="Capacity", value="68", unit="kWh", source="", action="write",
        band="auto", reason="", path="X", model_type="Property",
    )
    assert value_for_write(decision) == "68"


def test_a_created_range_is_built_as_a_range() -> None:
    decision = Decision(
        entity="Temp", value="-20...60", unit="", source="doc p.1", action="create",
        band="hosted", reason="", path="Technical", model_type="Range",
        id_short="Temp", value_type="xs:integer", min_value="-20", max_value="60",
    )
    element = element_for_create(decision)
    assert element["modelType"] == "Range"
    assert (element["min"], element["max"]) == ("-20", "60")
    assert element["valueType"] == "xs:integer"


def test_a_created_language_property_uses_the_element_shape() -> None:
    """POSTing a whole element is the other serialisation from PATCHing a value."""
    decision = Decision(
        entity="Designation", value="Motor@en", unit="", source="", action="create",
        band="hosted", reason="", path="Nameplate",
        model_type="MultiLanguageProperty", id_short="Designation",
    )
    assert element_for_create(decision)["value"] == [{"language": "en", "text": "Motor"}]


def test_a_created_list_child_is_built_without_an_id_short() -> None:
    decision = Decision(
        entity="Marking", value="CE", unit="", source="", action="create",
        band="hosted", reason="", path="Markings", model_type="Property",
        id_short=None, value_type="xs:string",
    )
    assert "idShort" not in element_for_create(decision)


def test_a_plan_written_before_types_were_decided_still_loads() -> None:
    """`model_type` used to be the only type field, and creates were Property."""
    plan = Plan.from_json(
        {
            "submodel": "urn:sm",
            "thresholds": {"high": 0.8, "low": 0.5},
            "decisions": [
                {"entity": "A", "value": "1", "unit": "", "source": "", "action": "create",
                 "band": "hosted", "reason": "", "path": "Technical",
                 "model_type": "Property", "value_type": "xs:integer", "id_short": "A"}
            ],
        }
    )
    assert plan.decisions[0].min_value is None
    assert element_for_create(plan.decisions[0])["modelType"] == "Property"


def test_the_type_fields_round_trip_through_a_plan() -> None:
    plan = Plan(submodel_id="urn:sm", theta_high=0.8, theta_low=0.5, model=None)
    plan.decisions.append(
        Decision(entity="Temp", value="-20...60", unit="", source="", action="create",
                 band="hosted", reason="", path="T", model_type="Range",
                 min_value="-20", max_value="60", content_type=None)
    )
    back = Plan.from_json(plan.to_json())
    assert (back.decisions[0].model_type, back.decisions[0].min_value) == ("Range", "-20")


def test_a_write_onto_a_container_is_refused_before_it_is_sent() -> None:
    """BaSyx answers this with a bare 500 that names no reason, and an agent
    reading that 500 has nothing to act on."""
    decision = Decision(
        entity="Stroke", value="300", unit="mm", source="", action="write",
        band="auto", reason="", path="Technical.Nozzle",
        model_type="SubmodelElementCollection",
    )
    with pytest.raises(MatchError, match="holds no value"):
        value_for_write(decision)


def test_writing_to_an_operation_says_it_is_invoked_instead() -> None:
    decision = Decision(
        entity="Speed", value="5", unit="", source="", action="write",
        band="auto", reason="", path="Ops.SetSpeed", model_type="Operation",
    )
    with pytest.raises(MatchError, match="invoked"):
        value_for_write(decision)


def test_a_hosted_decision_that_became_a_write_is_caught_by_its_band() -> None:
    """`hosted` is set by the create branch and by nothing else, so a write
    carrying it is a plan mangled between being made and being applied. The
    mangling also rewrote the type to Property, so the band is the only
    evidence left."""
    decision = Decision(
        entity="Base length", value="4522", unit="", source="", action="write",
        band="hosted", reason="", path="Technical.BasicData",
        model_type="Property",
    )
    with pytest.raises(MatchError, match="home"):
        value_for_write(decision)
