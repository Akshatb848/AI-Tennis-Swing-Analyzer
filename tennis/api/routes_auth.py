"""
Auth routes — JWT + Google OAuth authentication for TennisIQ.
"""

from __future__ import annotations

import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt

from tennis.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory user store (swap for DB in production) ────────────────────────
_users: dict[str, dict] = {}          # email -> user record
_users_by_id: dict[str, dict] = {}    # id -> user record


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    name: str
    subscription_tier: str


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def _create_or_get_user(email: str, name: str, auth_provider: str = "email") -> dict:
    """Get existing user or create a new one."""
    email = email.lower().strip()
    if email in _users:
        return _users[email]

    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "name": name or email.split("@")[0],
        "salt": secrets.token_hex(16),
        "password_hash": "",
        "subscription_tier": "free",
        "auth_provider": auth_provider,
        "created_at": datetime.utcnow().isoformat(),
    }
    _users[email] = user
    _users_by_id[user["id"]] = user
    return user


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest):
    """Register a new user account with email + password."""
    email = req.email.lower().strip()
    if email in _users:
        raise HTTPException(status_code=409, detail="Email already registered")

    salt = secrets.token_hex(16)
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "name": req.name or email.split("@")[0],
        "salt": salt,
        "password_hash": _hash_password(req.password, salt),
        "subscription_tier": "free",
        "auth_provider": "email",
        "created_at": datetime.utcnow().isoformat(),
    }
    _users[email] = user
    _users_by_id[user["id"]] = user

    token = _create_token(user["id"], email)
    logger.info("New user registered: %s", email)
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=email,
        name=user["name"],
        subscription_tier=user["subscription_tier"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Login with email + password and receive a JWT."""
    email = req.email.lower().strip()
    user = _users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.get("auth_provider") == "google" and not user.get("password_hash"):
        raise HTTPException(
            status_code=401,
            detail="This account uses Google sign-in. Please use 'Continue with Google'.",
        )

    expected = _hash_password(req.password, user["salt"])
    if expected != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_token(user["id"], email)
    logger.info("User logged in: %s", email)
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=email,
        name=user["name"],
        subscription_tier=user["subscription_tier"],
    )


@router.post("/google", response_model=TokenResponse)
async def google_auth(req: GoogleAuthRequest):
    """
    Authenticate with Google OAuth.

    Accepts a Google ID token, verifies it, and returns a JWT.
    Creates the user account if it doesn't exist.
    """
    try:
        # Try to verify with google-auth library
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        client_id = settings.GOOGLE_CLIENT_ID
        if not client_id:
            raise HTTPException(
                status_code=500,
                detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID in environment.",
            )

        idinfo = google_id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            client_id,
        )

        email = idinfo.get("email", "")
        name = idinfo.get("name", "")

        if not email:
            raise HTTPException(status_code=400, detail="Google token does not contain email")

    except ImportError:
        # google-auth not installed — decode JWT without verification (dev mode)
        logger.warning("google-auth not installed — using unverified token decode (dev only)")
        try:
            idinfo = jwt.decode(req.id_token, options={"verify_signature": False})
            email = idinfo.get("email", "")
            name = idinfo.get("name", idinfo.get("given_name", ""))
            if not email:
                raise HTTPException(status_code=400, detail="Token does not contain email")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {e}")

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Google token verification failed: {e}")

    # Create or get user
    user = _create_or_get_user(email, name, auth_provider="google")
    token = _create_token(user["id"], email)

    logger.info("Google auth: %s (%s)", email, "existing" if user.get("password_hash") else "new")
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=email,
        name=user["name"],
        subscription_tier=user["subscription_tier"],
    )


@router.get("/me")
async def get_me(user_id: str):
    """Get current user info (pass user_id from decoded JWT)."""
    user = _users_by_id.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "subscription_tier": user["subscription_tier"],
        "auth_provider": user.get("auth_provider", "email"),
    }
