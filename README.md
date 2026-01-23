OpenHCult

Local API smoke tests (pytest)

- Requires a running `hcultctrl` instance (local or remote).
- Default base URL: `http://127.0.0.1:8000`
- Override with `OPENHCULT_BASE_URL`

Example:

```bash
OPENHCULT_BASE_URL=http://192.168.0.117:8000 pytest -q
```
