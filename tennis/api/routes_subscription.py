"""
Subscription routes — Tier management, feature flags, entitlements.
"""

from __future__ import annotations

from fastapi import APIRouter

from tennis.models.subscription import (
    SUBSCRIPTION_PLANS,
    SubscriptionTier,
    UserEntitlement,
)

router = APIRouter()

_entitlements: dict[str, UserEntitlement] = {}


@router.get("/plans")
async def get_plans():
    """Get available subscription plans."""
    return [p.model_dump() for p in SUBSCRIPTION_PLANS]


@router.get("/user/{user_id}")
async def get_user_entitlement(user_id: str):
    """Get user's current subscription and entitlements."""
    ent = _entitlements.get(user_id)
    if not ent:
        ent = UserEntitlement(user_id=user_id)
        ent.refresh_features()
        _entitlements[user_id] = ent
    return ent.model_dump()


@router.post("/user/{user_id}/upgrade")
async def upgrade_subscription(user_id: str, tier: SubscriptionTier):
    """Upgrade user subscription tier."""
    ent = _entitlements.get(user_id)
    if not ent:
        ent = UserEntitlement(user_id=user_id)
        _entitlements[user_id] = ent
    ent.tier = tier
    ent.refresh_features()
    return {"user_id": user_id, "tier": tier, "features": ent.features}


@router.get("/user/{user_id}/can-start-session")
async def can_start_session(user_id: str):
    """Check if user can start a new session."""
    ent = _entitlements.get(user_id)
    if not ent:
        ent = UserEntitlement(user_id=user_id)
        ent.refresh_features()
        _entitlements[user_id] = ent
    return {
        "can_start": ent.can_start_session(),
        "sessions_used": ent.sessions_used_this_month,
        "sessions_limit": ent.sessions_limit,
        "tier": ent.tier,
    }


@router.get("/user/{user_id}/feature/{feature_name}")
async def check_feature(user_id: str, feature_name: str):
    """Check if user has access to a feature."""
    ent = _entitlements.get(user_id)
    if not ent:
        ent = UserEntitlement(user_id=user_id)
        ent.refresh_features()
        _entitlements[user_id] = ent
    return {
        "feature": feature_name,
        "has_access": ent.has_feature(feature_name),
        "tier": ent.tier,
    }
