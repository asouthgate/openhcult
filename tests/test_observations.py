import uuid
from test_utils import request_json
from urllib.error import HTTPError
import pytest


def test_observation_without_plant():
    note = f"obs-no-plant-{uuid.uuid4().hex[:8]}"
    created = request_json("/observations", method="POST", payload={"note": note})
    assert "id" in created
    assert created["note"] == note
    assert created["plant_name"] is None

    listed = request_json("/observations?limit=10000")
    found = next((r for r in listed["data"] if r["note"] == note), None)
    assert found is not None
    assert found["plant_name"] is None


def test_observation_with_plant():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    note = f"obs-with-plant-{uuid.uuid4().hex[:8]}"
    created = request_json(
        "/observations",
        method="POST",
        payload={"note": note, "plant_name": plant_name},
    )
    assert created["plant_name"] == plant_name

    listed = request_json("/observations?limit=10000")
    found = next((r for r in listed["data"] if r["note"] == note), None)
    assert found is not None
    assert found["plant_name"] == plant_name


def test_observation_unknown_plant_returns_404():
    with pytest.raises(HTTPError) as exc_info:
        request_json(
            "/observations",
            method="POST",
            payload={"note": "test", "plant_name": "nonexistent-plant-xyz"},
        )
    assert exc_info.value.code == 404


def test_delete_observation():
    note = f"obs-delete-{uuid.uuid4().hex[:8]}"
    created = request_json("/observations", method="POST", payload={"note": note})
    obs_id = created["id"]

    deleted = request_json(f"/observations/{obs_id}", method="DELETE")
    assert deleted["id"] == obs_id

    listed = request_json("/observations?limit=10000")
    ids = [r["id"] for r in listed["data"]]
    assert obs_id not in ids


def test_patch_observation_plant_name():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    note = f"obs-patch-{uuid.uuid4().hex[:8]}"
    created = request_json("/observations", method="POST", payload={"note": note})
    obs_id = created["id"]

    patched = request_json(
        f"/observations/{obs_id}",
        method="PATCH",
        payload={"plant_name": plant_name},
    )
    assert patched["plant_name"] == plant_name


def test_list_observations_filtered_by_plant():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_a = f"pytest-plant-a-{uuid.uuid4().hex[:8]}"
    plant_b = f"pytest-plant-b-{uuid.uuid4().hex[:8]}"
    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_a, "species_name": species_name},
    )
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_b, "species_name": species_name},
    )

    note_a = f"obs-filter-a-{uuid.uuid4().hex[:8]}"
    note_b = f"obs-filter-b-{uuid.uuid4().hex[:8]}"
    request_json(
        "/observations",
        method="POST",
        payload={"note": note_a, "plant_name": plant_a},
    )
    request_json(
        "/observations",
        method="POST",
        payload={"note": note_b, "plant_name": plant_b},
    )

    filtered_a = request_json(f"/observations?plant={plant_a}&limit=10000")
    notes_a = [r["note"] for r in filtered_a["data"]]
    assert note_a in notes_a
    assert note_b not in notes_a

    filtered_b = request_json(f"/observations?plant={plant_b}&limit=10000")
    notes_b = [r["note"] for r in filtered_b["data"]]
    assert note_b in notes_b
    assert note_a not in notes_b

    all_obs = request_json("/observations?limit=10000")
    all_notes = [r["note"] for r in all_obs["data"]]
    assert note_a in all_notes
    assert note_b in all_notes
