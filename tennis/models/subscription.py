"""
Subscription data models — Tiers, entitlements, and feature flags.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from tennis.timeutil import utcnow


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ELITE = "elite"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    TRIAL = "trial"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    GRACE_PERIOD = "grace_period"


class PaymentProvider(str, Enum):
    APPLE = "apple"
    GOOGLE = "google"
    STRIPE = "stripe"
    PROMO = "promo"


# ── Feature Flags ────────────────────────────────────────────────────────────

TIER_FEATURES: dict[SubscriptionTier, dict[str, bool | int]] = {
    SubscriptionTier.FREE: {
        "sessions_per_month": 3,
        "video_watermark": True,
        "cloud_processing": False,
        "full_stats": False,
        "shot_heatmaps": False,
        "highlights_auto": False,
        "coaching_feedback": False,
        "swing_analysis": False,
        "player_embedding": False,
        "weekly_goals": False,
        "remote_coach_access": False,
        "export_full_match": False,
        "export_highlights": True,
        "line_challenge": True,
        "live_streaming": False,
        "api_access": False,
        "priority_support": False,
    },
    SubscriptionTier.PRO: {
        "sessions_per_month": -1,  # unlimited
        "video_watermark": False,
        "cloud_processing": True,
        "full_stats": True,
        "shot_heatmaps": True,
        "highlights_auto": True,
        "coaching_feedback": False,
        "swing_analysis": False,
        "player_embedding": False,
        "weekly_goals": False,
        "remote_coach_access": False,
        "export_full_match": True,
        "export_highlights": True,
        "line_challenge": True,
        "live_streaming": True,
        "api_access": False,
        "priority_support": False,
    },
    SubscriptionTier.ELITE: {
        "sessions_per_month": -1,
        "video_watermark": False,
        "cloud_processing": True,
        "full_stats": True,
        "shot_heatmaps": True,
        "highlights_auto": True,
        "coaching_feedback": True,
        "swing_analysis": True,
        "player_embedding": True,
        "weekly_goals": True,
        "remote_coach_access": True,
        "export_full_match": True,
        "export_highlights": True,
        "line_challenge": True,
        "live_streaming": True,
        "api_access": True,
        "priority_support": True,
    },
}


class UserEntitlement(BaseModel):
    """Active subscription and feature entitlements for a user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    tier: SubscriptionTier = SubscriptionTier.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    payment_provider: PaymentProvider | None = None

    # ── Billing ──────────────────────────────────────────
    started_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    trial_ends_at: datetime | None = None
    price_monthly: float = 0.0
    currency: str = "USD"
    apple_receipt_id: str | None = None
    stripe_subscription_id: str | None = None

    # ── Usage tracking ───────────────────────────────────
    sessions_used_this_month: int = 0
    sessions_limit: int = 3
    cloud_storage_used_gb: float = 0.0
    cloud_storage_limit_gb: float = 1.0

    # ── Feature flags (computed from tier) ────────────────
    features: dict[str, bool | int] = Field(default_factory=dict)

    def refresh_features(self) -> None:
        """Refresh feature flags from tier definition."""
        self.features = dict(TIER_FEATURES.get(self.tier, TIER_FEATURES[SubscriptionTier.FREE]))
        self.sessions_limit = self.features.get("sessions_per_month", 3)

    def has_feature(self, feature_name: str) -> bool:
        """Check if user has access to a specific feature."""
        if not self.features:
            self.refresh_features()
        val = self.features.get(feature_name, False)
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return val != 0
        return False

    def can_start_session(self) -> bool:
        """Check if user can start a new session this month."""
        if self.sessions_limit == -1:
            return True
        return self.sessions_used_this_month < self.sessions_limit


class SubscriptionPlan(BaseModel):
    """A subscription plan available for purchase."""
    tier: SubscriptionTier
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: list[str]
    is_popular: bool = False
    apple_product_id: str | None = None
    google_product_id: str | None = None
    stripe_price_id: str | None = None


# ── Pre-defined plans ────────────────────────────────────────────────────────

SUBSCRIPTION_PLANS: list[SubscriptionPlan] = [
    SubscriptionPlan(
        tier=SubscriptionTier.FREE,
        name="Free",
        description="Get started with basic match tracking",
        price_monthly=0.0,
        price_yearly=0.0,
        features=[
            "3 sessions per month",
            "Basic scoring",
            "Line challenge system",
            "Highlight clips (watermarked)",
        ],
    ),
    SubscriptionPlan(
        tier=SubscriptionTier.PRO,
        name="Pro",
        description="Unlimited sessions with full stats & cloud processing",
        price_monthly=9.99,
        price_yearly=99.99,
        is_popular=True,
        features=[
            "Unlimited sessions",
            "No watermarks",
            "Cloud video processing",
            "Full match statistics",
            "Shot placement heatmaps",
            "Auto-generated highlights",
            "Full match export",
            "Live streaming",
        ],
    ),
    SubscriptionPlan(
        tier=SubscriptionTier.ELITE,
        name="Elite",
        description="AI coaching, swing analysis & remote coach access",
        price_monthly=24.99,
        price_yearly=249.99,
        features=[
            "Everything in Pro",
            "AI swing analysis",
            "Personalized coaching feedback",
            "Player style embedding",
            "Weekly adaptive goals",
            "Remote coach access",
            "API access",
            "Priority support",
        ],
    ),
]
