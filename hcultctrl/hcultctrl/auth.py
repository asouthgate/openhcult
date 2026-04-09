import json
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from hcultctrl import config

_bearer = HTTPBearer(auto_error=False)


def is_configured() -> bool:
    return config.get_credentials_path().exists()


def _load_credentials() -> dict:
    return json.loads(config.get_credentials_path().read_text())


def save_credentials(username: str, password: str) -> None:
    path = config.get_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    creds = {
        "username": username,
        "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        "jwt_secret": secrets.token_hex(32),
    }
    path.write_text(json.dumps(creds))
    os.chmod(path, 0o600)


def authenticate_user(username: str, password: str) -> bool:
    creds = _load_credentials()
    return username == creds["username"] and bcrypt.checkpw(
        password.encode(), creds["password_hash"].encode()
    )


def create_token(username: str) -> str:
    secret = _load_credentials()["jwt_secret"]
    expiry_hours = config.get_auth_token_expiry_hours()
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not configured"
        )
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    try:
        jwt.decode(
            credentials.credentials,
            _load_credentials()["jwt_secret"],
            algorithms=["HS256"],
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
