from fastapi.testclient import TestClient


def test_device_name_patch_calls_update(monkeypatch):
    from hcultctrl import api as ctrl_api

    called = {}

    def fake_update(conn, *, address, name):
        called["conn"] = conn
        called["address"] = address
        called["name"] = name

    class DummyConn:
        pass

    def override_get_db_conn():
        yield DummyConn()

    monkeypatch.setattr(ctrl_api.database, "update_device_name", fake_update)
    ctrl_api.app.dependency_overrides[ctrl_api._get_db_conn] = override_get_db_conn
    try:
        client = TestClient(ctrl_api.app)
        resp = client.patch("/devices/AA:BB:CC:DD:EE:FF", json={"name": "My Device"})
        assert resp.status_code == 200
        assert resp.json() == {
            "address": "AA:BB:CC:DD:EE:FF",
            "name": "My Device",
        }
        assert called["address"] == "AA:BB:CC:DD:EE:FF"
        assert called["name"] == "My Device"
        assert isinstance(called["conn"], DummyConn)
    finally:
        ctrl_api.app.dependency_overrides.pop(ctrl_api._get_db_conn, None)
