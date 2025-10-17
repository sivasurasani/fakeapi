from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field


UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


app = FastAPI(
    title="Fake API Orchestrator",
    version="0.1.0",
    description="In-memory mock services for local development while real APIs are implemented.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"


@dataclass
class SessionRecord:
    session_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    @property
    def status(self) -> SessionStatus:
        return (
            SessionStatus.ACTIVE
            if self.expires_at > utcnow()
            else SessionStatus.EXPIRED
        )


class SessionManager:
    """Tracks anonymous session lifecycles with transparent regeneration."""

    def __init__(self, ttl_minutes: int = 30, regen_threshold_minutes: int = 5) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._regen_threshold = timedelta(minutes=regen_threshold_minutes)
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def create_or_refresh(
        self, candidate_session_id: Optional[str]
    ) -> Tuple[SessionRecord, Optional[str], bool]:
        """Create a new session or refresh an existing one.

        Returns a tuple of (current record, previous session id, regenerated flag).
        """
        async with self._lock:
            now = utcnow()
            record: Optional[SessionRecord] = None
            previous_id: Optional[str] = None

            if candidate_session_id:
                record = self._sessions.get(candidate_session_id)
                if record and record.expires_at <= now:
                    # Expired sessions are discarded.
                    self._sessions.pop(candidate_session_id, None)
                    record = None
                elif record:
                    previous_id = candidate_session_id

            regenerated = False

            if record:
                lifetime_remaining = record.expires_at - now
                if lifetime_remaining <= self._regen_threshold:
                    # Regenerate to mitigate fixation and prolong the session.
                    new_id = self._generate_session_id()
                    new_record = SessionRecord(
                        session_id=new_id,
                        created_at=record.created_at,
                        updated_at=now,
                        expires_at=now + self._ttl,
                    )
                    self._sessions[new_id] = new_record
                    self._sessions.pop(record.session_id, None)
                    record = new_record
                    regenerated = True
                else:
                    record.updated_at = now
                    record.expires_at = now + self._ttl
            else:
                new_id = self._generate_session_id()
                record = SessionRecord(
                    session_id=new_id,
                    created_at=now,
                    updated_at=now,
                    expires_at=now + self._ttl,
                )
                self._sessions[new_id] = record
                regenerated = True if candidate_session_id else False

            return record, previous_id, regenerated

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record and record.expires_at > utcnow():
                return record
            if record:
                self._sessions.pop(session_id, None)
            return None

    @staticmethod
    def _generate_session_id() -> str:
        return uuid.uuid4().hex


session_manager = SessionManager()


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Existing session identifier to refresh. If omitted, a new session is created.",
        json_schema_extra={"example": "9f10c8e4c3c94d0b8a3f1e6f72ab1234"},
    )


class SessionResponse(BaseModel):
    session_id: str
    expires_at: datetime
    regenerated: bool
    previous_session_id: Optional[str]


@app.post("/api/sessions", response_model=SessionResponse, tags=["Session Management"])
async def upsert_session(
    payload: SessionRequest = Body(
        default=SessionRequest(session_id=None),
        example={"session_id": "existing-session-id-1234"},
        description="Provide an existing session_id to refresh, or omit to create a new session.",
    ),
) -> SessionResponse:
    record, previous_id, regenerated = await session_manager.create_or_refresh(
        payload.session_id
    )
    return SessionResponse(
        session_id=record.session_id,
        expires_at=record.expires_at,
        regenerated=regenerated,
        previous_session_id=previous_id,
    )


# ---------------------------------------------------------------------------
# Account and profile lifecycle
# ---------------------------------------------------------------------------


class RegistrationStatus(str, Enum):
    OTP_SENT = "OTP_SENT"
    OTP_RESENT = "OTP_RESENT"
    ACCOUNT_EXISTS = "ACCOUNT_EXISTS"
    TRIAL_LIMIT_EXCEEDED = "TRIAL_LIMIT_EXCEEDED"


class Mode(str, Enum):
    DEMO = "demo"
    LIVE = "live"


@dataclass
class AccountRecord:
    email: EmailStr
    token: str
    otp_code: str
    otp_expires_at: datetime
    verified: bool = False
    profile_key: Optional[str] = None
    devices: Dict[Mode, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def refresh_otp(self, ttl_minutes: int) -> None:
        self.otp_code = f"{random.randint(0, 999_999):06d}"
        self.otp_expires_at = utcnow() + timedelta(minutes=ttl_minutes)
        self.updated_at = utcnow()


class AccountStore:
    def __init__(self, otp_ttl_minutes: int = 10) -> None:
        self._otp_ttl_minutes = otp_ttl_minutes
        self._accounts: Dict[str, AccountRecord] = {}
        self._profile_index: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, email: EmailStr
    ) -> Tuple[AccountRecord, RegistrationStatus]:
        async with self._lock:
            key = email.lower()
            record = self._accounts.get(key)
            if record and record.verified:
                if len(
                    {mode for mode, device in record.devices.items() if device}
                ) >= len(Mode):
                    return record, RegistrationStatus.TRIAL_LIMIT_EXCEEDED
                return record, RegistrationStatus.ACCOUNT_EXISTS

            if record:
                record.refresh_otp(self._otp_ttl_minutes)
                return record, RegistrationStatus.OTP_RESENT

            record = AccountRecord(
                email=email,
                token=uuid.uuid4().hex,
                otp_code=f"{random.randint(0, 999_999):06d}",
                otp_expires_at=utcnow() + timedelta(minutes=self._otp_ttl_minutes),
            )
            self._accounts[key] = record
            return record, RegistrationStatus.OTP_SENT

    async def verify_otp(
        self, email: EmailStr, otp_code: str, token: str
    ) -> AccountRecord:
        async with self._lock:
            key = email.lower()
            record = self._accounts.get(key)
            if not record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Account not found."
                )
            now = utcnow()
            if record.token != token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token."
                )
            if record.otp_expires_at <= now:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="OTP expired."
                )
            if record.otp_code != otp_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP code."
                )

            record.verified = True
            if not record.profile_key:
                record.profile_key = uuid.uuid4().hex
                self._profile_index[record.profile_key] = key
            record.updated_at = now
            # Once verified, invalidate OTP to enforce single use.
            record.otp_code = f"{random.randint(0, 999_999):06d}"
            record.otp_expires_at = now  # mark as expired/consumed
            return record

    async def get_by_profile_key(self, profile_key: str) -> Optional[AccountRecord]:
        async with self._lock:
            email_key = self._profile_index.get(profile_key)
            if not email_key:
                return None
            return self._accounts.get(email_key)

    async def get(self, email: EmailStr) -> Optional[AccountRecord]:
        async with self._lock:
            return self._accounts.get(email.lower())


account_store = AccountStore()


class TrialRegistrationRequest(BaseModel):
    email: EmailStr = Field(..., json_schema_extra={"example": "trial.user@example.com"})


class TrialRegistrationResponse(BaseModel):
    status: RegistrationStatus
    message: str = Field(
        ...,
        json_schema_extra={"example": "OTP code generated. Complete verification to continue onboarding."},
    )
    token: Optional[str] = Field(None, json_schema_extra={"example": "3d2e1c4b5a6f7890"})
    otp_code: Optional[str] = Field(
        None,
        description="Mock OTP code surfaced for local testing only. Real systems should omit this.",
        json_schema_extra={"example": "123456"},
    )
    otp_expires_at: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2024-10-17T15:42:36.780Z"},
    )


@app.post(
    "/api/accounts/trial",
    response_model=TrialRegistrationResponse,
    tags=["Account Registration"],
)
async def register_trial_account(
    payload: TrialRegistrationRequest = Body(
        ...,
        example={"email": "trial.user@example.com"},
        description="Email address for the trial account registration request.",
    ),
) -> TrialRegistrationResponse:
    record, status_code = await account_store.register(payload.email)
    message = {
        RegistrationStatus.OTP_SENT: "OTP code generated. Complete verification to continue onboarding.",
        RegistrationStatus.OTP_RESENT: "Existing trial detected. A fresh OTP has been issued.",
        RegistrationStatus.ACCOUNT_EXISTS: "Account already active. Please sign in.",
        RegistrationStatus.TRIAL_LIMIT_EXCEEDED: "Trial device limit reached.",
    }[status_code]

    return TrialRegistrationResponse(
        status=status_code,
        message=message,
        token=record.token
        if status_code in (RegistrationStatus.OTP_SENT, RegistrationStatus.OTP_RESENT)
        else None,
        otp_code=record.otp_code
        if status_code in (RegistrationStatus.OTP_SENT, RegistrationStatus.OTP_RESENT)
        else None,
        otp_expires_at=record.otp_expires_at
        if status_code in (RegistrationStatus.OTP_SENT, RegistrationStatus.OTP_RESENT)
        else None,
    )


class OTPValidationRequest(BaseModel):
    email: EmailStr = Field(..., json_schema_extra={"example": "trial.user@example.com"})
    otp_code: str = Field(
        ...,
        min_length=4,
        max_length=6,
        json_schema_extra={"example": "123456"},
    )
    token: str = Field(..., json_schema_extra={"example": "3d2e1c4b5a6f7890"})


class OTPValidationResponse(BaseModel):
    profile_key: str = Field(
        ...,
        json_schema_extra={"example": "f2d3c4b5a6978877665544332211eeff"},
    )
    message: str = Field(
        ...,
        json_schema_extra={"example": "OTP verified. Profile ready for next steps."},
    )


@app.post(
    "/api/accounts/otp/verify",
    response_model=OTPValidationResponse,
    tags=["Account Registration"],
)
async def validate_otp(
    payload: OTPValidationRequest = Body(
        ...,
        example={
            "email": "trial.user@example.com",
            "otp_code": "123456",
            "token": "3d2e1c4b5a6f7890",
        },
        description="Submit the OTP and issued token to complete verification.",
    ),
) -> OTPValidationResponse:
    record = await account_store.verify_otp(
        payload.email, payload.otp_code, payload.token
    )
    return OTPValidationResponse(
        profile_key=record.profile_key or "",
        message="OTP verified. Profile ready for next steps.",
    )


# ---------------------------------------------------------------------------
# Validation lifecycle
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    EXPIRED = "expired"


@dataclass
class ValidationRecord:
    validation_id: str
    topic_id: str
    profile_key: str
    chat_id: str
    status: ValidationStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    history: List[Tuple[datetime, ValidationStatus, str]] = field(default_factory=list)

    def advance(self) -> None:
        now = utcnow()
        if self.status in (
            ValidationStatus.SUCCESS,
            ValidationStatus.FAILURE,
            ValidationStatus.EXPIRED,
        ):
            return

        elapsed = (now - self.created_at).total_seconds()
        if elapsed > 60:
            self.status = ValidationStatus.EXPIRED
            self.history.append((now, self.status, "Validation expired."))
        elif elapsed > 15:
            self.status = ValidationStatus.SUCCESS
            self.history.append((now, self.status, "Validation successful."))
        elif elapsed > 5 and self.status == ValidationStatus.IN_PROGRESS:
            # Simulate mid-validation update
            self.history.append((now, self.status, "Connectivity checks in progress."))
        self.updated_at = now


class ValidationEngine:
    def __init__(self) -> None:
        self._records: Dict[str, ValidationRecord] = {}
        self._lock = asyncio.Lock()

    async def start(self, profile_key: str, chat_id: str) -> ValidationRecord:
        async with self._lock:
            validation_id = uuid.uuid4().hex
            topic_id = uuid.uuid4().hex
            now = utcnow()
            record = ValidationRecord(
                validation_id=validation_id,
                topic_id=topic_id,
                profile_key=profile_key,
                chat_id=chat_id,
                status=ValidationStatus.IN_PROGRESS,
                created_at=now,
                updated_at=now,
                expires_at=now + timedelta(minutes=10),
            )
            record.history.append(
                (now, ValidationStatus.IN_PROGRESS, "Validation started.")
            )
            self._records[validation_id] = record
            return record

    async def get(self, validation_id: str) -> Optional[ValidationRecord]:
        async with self._lock:
            record = self._records.get(validation_id)
            if not record:
                return None
            record.advance()
            return record


validation_engine = ValidationEngine()


class ValidationStartRequest(BaseModel):
    profile_key: str = Field(
        ...,
        json_schema_extra={"example": "f2d3c4b5a6978877665544332211eeff"},
    )
    chat_id: str = Field(..., json_schema_extra={"example": "chat-12345"})


class ValidationStartResponse(BaseModel):
    validation_id: str = Field(
        ...,
        json_schema_extra={"example": "val-9c3d2b1a0f"},
    )
    topic_id: str = Field(..., json_schema_extra={"example": "topic-555"})
    status: ValidationStatus


@app.post(
    "/api/validation/start",
    response_model=ValidationStartResponse,
    tags=["Validation"],
)
async def start_validation(
    payload: ValidationStartRequest = Body(
        ...,
        example={
            "profile_key": "f2d3c4b5a6978877665544332211eeff",
            "chat_id": "chat-12345",
        },
        description="Kick off validation for the profile-to-chat pairing.",
    ),
) -> ValidationStartResponse:
    account = await account_store.get_by_profile_key(payload.profile_key)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found."
        )
    record = await validation_engine.start(
        profile_key=payload.profile_key, chat_id=payload.chat_id
    )
    return ValidationStartResponse(
        validation_id=record.validation_id,
        topic_id=record.topic_id,
        status=record.status,
    )


class ValidationProgress(BaseModel):
    timestamp: datetime = Field(
        ...,
        json_schema_extra={"example": "2024-10-17T15:42:41.000Z"},
    )
    status: ValidationStatus
    message: str = Field(
        ...,
        json_schema_extra={"example": "Connectivity checks in progress."},
    )


class ValidationStatusResponse(BaseModel):
    validation_id: str = Field(
        ...,
        json_schema_extra={"example": "val-9c3d2b1a0f"},
    )
    status: ValidationStatus
    topic_id: str = Field(..., json_schema_extra={"example": "topic-555"})
    last_updated: datetime = Field(
        ...,
        json_schema_extra={"example": "2024-10-17T15:42:50.000Z"},
    )
    expires_at: datetime = Field(
        ...,
        json_schema_extra={"example": "2024-10-17T15:52:36.780Z"},
    )
    history: List[ValidationProgress]


@app.get(
    "/api/validation/status/{validation_id}",
    response_model=ValidationStatusResponse,
    tags=["Validation"],
)
async def get_validation_status(validation_id: str) -> ValidationStatusResponse:
    record = await validation_engine.get(validation_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Validation not found."
        )

    history = [
        ValidationProgress(timestamp=ts, status=status, message=message)
        for ts, status, message in record.history
    ]
    return ValidationStatusResponse(
        validation_id=record.validation_id,
        status=record.status,
        topic_id=record.topic_id,
        last_updated=record.updated_at,
        expires_at=record.expires_at,
        history=history,
    )


# ---------------------------------------------------------------------------
# Container orchestration lifecycle
# ---------------------------------------------------------------------------


class ContainerStatus(str, Enum):
    IN_PROGRESS = "in-progress"
    TRAINING = "training"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ContainerSpawnStatus(str, Enum):
    SUCCESS = "success"
    ALREADY_ACTIVE = "already_active"
    ERROR = "error"


DEFAULT_ENVIRONMENT = {
    "training_secs": 120,
    "cycle_counter": 0,
    "days_to_maintenance": 14,
}


@dataclass
class ContainerRecord:
    device_id: str
    profile_key: str
    mode: Mode
    session_id: str
    environment_config: Dict[str, int]
    status: ContainerStatus
    created_at: datetime
    updated_at: datetime
    demo_feed_active: bool = False

    def advance(self) -> None:
        now = utcnow()
        elapsed = (now - self.created_at).total_seconds()
        if self.status in (ContainerStatus.INACTIVE, ContainerStatus.ERROR):
            return
        if elapsed > 20 and self.status != ContainerStatus.ACTIVE:
            self.status = ContainerStatus.ACTIVE
        elif elapsed > 10 and self.status == ContainerStatus.IN_PROGRESS:
            self.status = ContainerStatus.TRAINING
        elif elapsed > 40:
            self.status = ContainerStatus.ERROR
        self.updated_at = now
        if self.mode == Mode.DEMO and self.status == ContainerStatus.ACTIVE:
            self.demo_feed_active = True


class ContainerOrchestrator:
    def __init__(self) -> None:
        self._containers: Dict[str, ContainerRecord] = {}
        self._lock = asyncio.Lock()

    async def spawn(
        self,
        *,
        account: AccountRecord,
        device_id: str,
        mode: Mode,
        session: SessionRecord,
        environment_config: Dict[str, int],
    ) -> Tuple[ContainerRecord, ContainerSpawnStatus, str]:
        async with self._lock:
            now = utcnow()
            normalized_device_id = device_id.lower()
            existing_device_id = account.devices.get(mode)

            if existing_device_id and existing_device_id != normalized_device_id:
                existing_container = self._containers.get(existing_device_id)
                if existing_container:
                    existing_container.advance()
                else:
                    existing_container = ContainerRecord(
                        device_id=existing_device_id,
                        profile_key=account.profile_key or "",
                        mode=mode,
                        session_id=session.session_id,
                        environment_config=environment_config,
                        status=ContainerStatus.INACTIVE,
                        created_at=now,
                        updated_at=now,
                    )
                return (
                    existing_container,
                    ContainerSpawnStatus.ERROR,
                    "Trial device limit exceeded for this mode.",
                )

            container = self._containers.get(normalized_device_id)
            if container and container.status != ContainerStatus.INACTIVE:
                container.advance()
                message = (
                    "Already Active"
                    if container.status == ContainerStatus.ACTIVE
                    else "Already Provisioning"
                )
                outcome = ContainerSpawnStatus.ALREADY_ACTIVE
                return container, outcome, message

            status_label = "Provisioning started"
            if not container:
                container = ContainerRecord(
                    device_id=normalized_device_id,
                    profile_key=account.profile_key or "",
                    mode=mode,
                    session_id=session.session_id,
                    environment_config=environment_config,
                    status=ContainerStatus.IN_PROGRESS,
                    created_at=now,
                    updated_at=now,
                    demo_feed_active=False,
                )
            else:
                container.status = ContainerStatus.IN_PROGRESS
                container.created_at = now
                container.updated_at = now
                container.environment_config = environment_config

            if mode == Mode.DEMO:
                container.demo_feed_active = True

            self._containers[normalized_device_id] = container
            account.devices[mode] = normalized_device_id
            return container, ContainerSpawnStatus.SUCCESS, status_label

    async def get(self, device_id: str) -> Optional[ContainerRecord]:
        async with self._lock:
            container = self._containers.get(device_id.lower())
            if container:
                container.advance()
            return container


container_orchestrator = ContainerOrchestrator()


class ContainerSpawnRequest(BaseModel):
    profile_key: str = Field(
        ...,
        json_schema_extra={"example": "f2d3c4b5a6978877665544332211eeff"},
    )
    mode: Mode
    device_id: str = Field(..., json_schema_extra={"example": "device-123"})
    session_id: str = Field(
        ...,
        json_schema_extra={"example": "9f10c8e4c3c94d0b8a3f1e6f72ab1234"},
    )
    training_secs: Optional[int] = Field(
        None,
        ge=10,
        json_schema_extra={"example": 180},
    )
    cycle_counter: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"example": 25},
    )
    days_to_maintenance: Optional[int] = Field(
        None,
        ge=1,
        json_schema_extra={"example": 7},
    )


class ContainerSpawnResponse(BaseModel):
    status: ContainerSpawnStatus
    device_id: str = Field(..., json_schema_extra={"example": "device-123"})
    health_state: ContainerStatus
    environment_config: Dict[str, int] = Field(
        ...,
        json_schema_extra={
            "example": {"training_secs": 180, "cycle_counter": 25, "days_to_maintenance": 7}
        },
    )
    message: str = Field(..., json_schema_extra={"example": "Provisioning started"})


@app.post(
    "/api/containers/spawn",
    response_model=ContainerSpawnResponse,
    tags=["Container"],
)
async def spawn_container(
    payload: ContainerSpawnRequest = Body(
        ...,
        example={
            "profile_key": "f2d3c4b5a6978877665544332211eeff",
            "mode": "demo",
            "device_id": "device-123",
            "session_id": "9f10c8e4c3c94d0b8a3f1e6f72ab1234",
            "training_secs": 180,
            "cycle_counter": 25,
            "days_to_maintenance": 7,
        },
        description="Provision or resume an MI container for the specified profile and device.",
    ),
) -> ContainerSpawnResponse:
    account = await account_store.get_by_profile_key(payload.profile_key)
    if not account or not account.verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profile not ready for container spawn.",
        )

    session = await session_manager.get(payload.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid or expired.",
        )

    environment_config = {
        "training_secs": payload.training_secs or DEFAULT_ENVIRONMENT["training_secs"],
        "cycle_counter": payload.cycle_counter
        if payload.cycle_counter is not None
        else DEFAULT_ENVIRONMENT["cycle_counter"],
        "days_to_maintenance": payload.days_to_maintenance
        or DEFAULT_ENVIRONMENT["days_to_maintenance"],
    }

    container, outcome, message = await container_orchestrator.spawn(
        account=account,
        device_id=payload.device_id,
        mode=payload.mode,
        session=session,
        environment_config=environment_config,
    )

    return ContainerSpawnResponse(
        status=outcome,
        device_id=container.device_id,
        health_state=container.status,
        environment_config=container.environment_config,
        message=message,
    )


class ContainerHealthResponse(BaseModel):
    device_id: str = Field(..., json_schema_extra={"example": "device-123"})
    status: ContainerStatus
    demo_feed_active: bool = Field(..., json_schema_extra={"example": True})
    last_updated: datetime = Field(
        ...,
        json_schema_extra={"example": "2024-10-17T15:43:16.780Z"},
    )


@app.get(
    "/api/containers/{device_id}/health",
    response_model=ContainerHealthResponse,
    tags=["Container"],
)
async def get_container_health(device_id: str) -> ContainerHealthResponse:
    container = await container_orchestrator.get(device_id)
    if not container:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Container not found."
        )
    return ContainerHealthResponse(
        device_id=container.device_id,
        status=container.status,
        demo_feed_active=container.demo_feed_active,
        last_updated=container.updated_at,
    )


@app.get("/health", tags=["Diagnostics"])
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mock-api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
