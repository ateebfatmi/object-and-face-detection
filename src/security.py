from __future__ import annotations

from dataclasses import dataclass
import os

import streamlit as st


@dataclass(frozen=True, slots=True)
class DeploymentSettings:
    environment: str
    allowed_emails: frozenset[str]
    retention_days: int | None
    auth_provider: str | None

    @property
    def production(self) -> bool:
        return self.environment in {"production", "prod"}

    @classmethod
    def from_environment(cls) -> "DeploymentSettings":
        raw_days = os.getenv("VISIONGUARD_ATTENDANCE_RETENTION_DAYS", "").strip()
        retention_days: int | None = None
        if raw_days:
            try:
                parsed = int(raw_days)
                retention_days = parsed if parsed > 0 else None
            except ValueError:
                retention_days = None
        emails = frozenset(
            item.strip().casefold()
            for item in os.getenv("VISIONGUARD_ALLOWED_EMAILS", "").split(",")
            if item.strip()
        )
        return cls(
            environment=os.getenv("VISIONGUARD_ENV", "development").strip().casefold(),
            allowed_emails=emails,
            retention_days=retention_days,
            auth_provider=os.getenv("VISIONGUARD_AUTH_PROVIDER", "").strip() or None,
        )

    def configuration_errors(self) -> list[str]:
        if not self.production:
            return []
        errors: list[str] = []
        if not self.allowed_emails:
            errors.append("VISIONGUARD_ALLOWED_EMAILS must contain at least one authorized account.")
        if self.retention_days is None:
            errors.append("VISIONGUARD_ATTENDANCE_RETENTION_DAYS must be a positive integer.")
        return errors


def current_actor() -> str:
    try:
        if st.user.is_logged_in:
            return str(st.user.get("email") or st.user.get("sub") or "authenticated-user")
    except Exception:
        pass
    return "local-development"


def enforce_production_access(settings: DeploymentSettings) -> None:
    """Fail closed in production until OIDC, an allowlist, and retention are configured."""
    errors = settings.configuration_errors()
    if errors:
        st.error("Production configuration is incomplete.")
        for error in errors:
            st.write(f"- {error}")
        st.caption("The app is locked to prevent unauthorized biometric-data access.")
        st.stop()
    if not settings.production:
        return

    try:
        logged_in = bool(st.user.is_logged_in)
    except Exception:
        st.error("Production authentication is not configured correctly.")
        st.write("Check the OIDC settings in `.streamlit/secrets.toml` and restart the app.")
        st.stop()

    if not logged_in:
        st.title("Sign in to VisionGuard")
        st.write("Attendance and biometric records are restricted to authorized staff.")
        if st.button("Sign in", type="primary"):
            try:
                st.login(settings.auth_provider)
            except Exception:
                st.error("Sign-in is not available. Check the configured OIDC provider and try again.")
        st.stop()

    email = str(st.user.get("email") or "").strip().casefold()
    if email not in settings.allowed_emails:
        st.error("Access denied")
        st.write("This account is signed in but is not authorized to access VisionGuard.")
        if st.button("Sign out"):
            st.logout()
        st.stop()


def show_identity_controls() -> None:
    settings = DeploymentSettings.from_environment()
    if not settings.production:
        return
    identity = str(st.user.get("email") or st.user.get("name") or "Authorized user")
    with st.sidebar:
        st.caption(f"Signed in as {identity}")
        if st.button("Sign out", key="visionguard_sign_out"):
            st.logout()
