import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from hcultctrl import config
from hcultdb.connection import connect
from hcultdb.queries import count_users, create_user, get_user_by_username

_bearer = HTTPBearer(auto_error=False)


def is_configured() -> bool:
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        return count_users(conn) > 0
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        user = get_user_by_username(conn, username=username)
        if user is None:
            return False
        return bcrypt.checkpw(password.encode(), user["password_hash"].encode())
    finally:
        conn.close()


def save_credentials(username: str, password: str) -> None:
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        jwt_secret = secrets.token_hex(32)
        create_user(
            conn, username=username, password_hash=password_hash, jwt_secret=jwt_secret
        )
    finally:
        conn.close()


def create_token(username: str) -> str:
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        user = get_user_by_username(conn, username=username)
        secret = user["jwt_secret"]
    finally:
        conn.close()
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
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        unverified = jwt.decode(
            credentials.credentials,
            options={"verify_signature": False, "verify_exp": True},
            algorithms=["HS256"],
        )
        username = unverified.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        user = get_user_by_username(conn, username=username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        jwt.decode(
            credentials.credentials,
            user["jwt_secret"],
            algorithms=["HS256"],
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    finally:
        conn.close()
