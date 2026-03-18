import uuid
from test_api import _request_json
from urllib.error import HTTPError
import pytest


def test_observation_without_plant():
    note = f"obs-no-plant-{uuid.uuid4().hex[:8]}"
    created = _request_json("/observations", method="POST", payload={"note": note})
    assert "id" in created
    assert created["note"] == note
    assert created["plant_name"] is None

    listed = _request_json("/observations?limit=10000")
    found = next((r for r in listed["data"] if r["note"] == note), None)
    assert found is not None
    assert found["plant_name"] is None


def test_observation_with_plant():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    _request_json("/species", method="POST", payload={"name": species_name})
    _request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    note = f"obs-with-plant-{uuid.uuid4().hex[:8]}"
    created = _request_json(
        "/observations",
        method="POST",
        payload={"note": note, "plant_name": plant_name},
    )
    assert created["plant_name"] == plant_name

    listed = _request_json("/observations?limit=10000")
    found = next((r for r in listed["data"] if r["note"] == note), None)
    assert found is not None
    assert found["plant_name"] == plant_name


def test_observation_unknown_plant_returns_404():
    with pytest.raises(HTTPError) as exc_info:
        _request_json(
            "/observations",
            method="POST",
            payload={"note": "test", "plant_name": "nonexistent-plant-xyz"},
        )
    assert exc_info.value.code == 404


def test_patch_observation_plant_name():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    _request_json("/species", method="POST", payload={"name": species_name})
    _request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    note = f"obs-patch-{uuid.uuid4().hex[:8]}"
    created = _request_json("/observations", method="POST", payload={"note": note})
    obs_id = created["id"]

    patched = _request_json(
        f"/observations/{obs_id}",
        method="PATCH",
        payload={"plant_name": plant_name},
    )
    assert patched["plant_name"] == plant_name
