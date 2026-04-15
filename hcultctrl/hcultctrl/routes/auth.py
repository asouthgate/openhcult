from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from hcultctrl.auth import (
    authenticate_user,
    create_token,
    is_configured,
    save_credentials,
)

router = APIRouter()


class Credentials(BaseModel):
    username: str
    password: str


@router.get("/auth/status")
def auth_status():
    return {"configured": is_configured()}


@router.post("/auth/setup")
def setup(payload: Credentials):
    if is_configured():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Already configured"
        )
    if not payload.username.strip() or not payload.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required",
        )
    save_credentials(payload.username.strip(), payload.password)
    return {
        "access_token": create_token(payload.username.strip()),
        "token_type": "bearer",
    }


@router.post("/auth/login")
def login(payload: Credentials):
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not configured"
        )
    if not authenticate_user(payload.username, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return {"access_token": create_token(payload.username), "token_type": "bearer"}
