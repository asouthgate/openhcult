from types import SimpleNamespace


def test_cli_devices_name_calls_ctrl(monkeypatch):
    from hcultutils import cli as hcult_cli

    called = {}

    def fake_request(method, url, payload=None):
        called["method"] = method
        called["url"] = url
        called["payload"] = payload
        return {"address": "AA:BB:CC:DD:EE:FF", "name": "My Device"}

    monkeypatch.setattr(hcult_cli, "_request_ctrl", fake_request)
    args = SimpleNamespace(address="AA:BB:CC:DD:EE:FF", name="My Device")
    status = hcult_cli._devices_via_ctrl("http://127.0.0.1:8000", "name", args)

    assert status == 0
    assert called["method"] == "PATCH"
    assert called["url"] == "http://127.0.0.1:8000/devices/AA%3ABB%3ACC%3ADD%3AEE%3AFF"
    assert called["payload"] == {"name": "My Device"}
