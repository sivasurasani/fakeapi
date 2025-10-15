from __future__ import annotations

import random
import string
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import json
from pathlib import Path

router = APIRouter(prefix="/mock", tags=["mock"])

# Paths for mock persistence
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MOCK_DATA_DIR = _PROJECT_ROOT / "mock_data"
_MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
_ACCOUNTS_DB_FILE = _MOCK_DATA_DIR / "accounts_db.json"
_SESSIONS_DB_FILE = _MOCK_DATA_DIR / "sessions_db.json"

# In-memory stores for mock state
_STATE: Dict[str, Any] = {
    "accounts": {},  # email -> {otp, is_premature, profile_key, password?, access_token?}
    "profiles": {},  # profile_key -> {name, owner_email}
    "devices": {},  # device_id -> {profile_key, mode, status, created_at, topic, username, password, broker}
    "trial_instances": {},  # trial_key -> count
    "sessions": {},  # session_id -> {email, created_at}
}


def _load_json_db(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json_db(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        # ignore persistence errors in mock
        pass


# Initialize from disk (idempotent)
_STATE["accounts"].update(_load_json_db(_ACCOUNTS_DB_FILE))
_STATE["sessions"].update(_load_json_db(_SESSIONS_DB_FILE))

# Constants
MAX_TRIAL_INSTANCES = 2


def _random_key(prefix: str, length: int = 12) -> str:
    return (
        prefix
        + "_"
        + "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


class RegisterRequest(BaseModel):
    email: str
    session_id: Optional[str] = None


class RegisterResponse(BaseModel):
    success: bool
    accountExists: bool = False
    shouldLogin: bool = False
    isPremature: bool = False
    otp: Optional[str] = None


@router.post("/register", response_model=RegisterResponse)
def register_account(req: RegisterRequest) -> RegisterResponse:
    # Accept any input - extract email if it looks like one, otherwise use a default
    email_input = req.email.strip().lower()

    # Simple email extraction - if contains @, use as is, otherwise create a dummy one
    if "@" in email_input:
        email = email_input
    else:
        # Use the input as a username with a dummy domain
        email = f"{email_input.replace(' ', '_')}@demo.local"

    # Heuristic: emails containing "exists" simulate existing verified accounts
    if "exists" in email:
        return RegisterResponse(
            success=True, accountExists=True, shouldLogin=True, isPremature=False
        )

    # Create or update a premature account and send an OTP
    otp = "123456"
    is_premature = True
    _STATE["accounts"].setdefault(email, {})
    _STATE["accounts"][email].update({"otp": otp, "is_premature": is_premature})
    return RegisterResponse(
        success=True,
        accountExists=False,
        shouldLogin=False,
        isPremature=is_premature,
        otp=otp,
    )


class ValidateRequest(BaseModel):
    email: str
    otp: str


class ValidateResponse(BaseModel):
    success: bool
    profileKey: str
    deviceId: Optional[str] = None


@router.post("/validate", response_model=ValidateResponse)
def validate_otp(req: ValidateRequest) -> ValidateResponse:
    email_input = req.email.strip().lower()
    otp_input = req.otp.strip()

    # Extract email if it looks like one
    if "@" in email_input:
        email = email_input
    else:
        email = f"{email_input.replace(' ', '_')}@demo.local"

    # Be lenient - if OTP is "123456" or the account doesn't exist yet, create it
    record = _STATE["accounts"].get(email)

    # Always succeed for demo purposes
    if not record:
        # Create account on the fly
        _STATE["accounts"][email] = {"otp": "123456", "is_premature": True}
        record = _STATE["accounts"][email]

    # Profile key may already exist (premature account). Do not recreate.
    profile_key = record.get("profile_key") or _random_key("profile")
    record["profile_key"] = profile_key
    _STATE["profiles"].setdefault(
        profile_key, {"name": f"Profile for {email}", "owner_email": email}
    )

    # Generate a placeholder device ID that will be replaced when device is spawned
    device_id_placeholder = _random_key("dev")

    return ValidateResponse(
        success=True, profileKey=profile_key, deviceId=device_id_placeholder
    )


class SpawnRequest(BaseModel):
    profileKey: str
    mode: str  # "demo" or "live"
    sessionId: Optional[str] = None
    sessionExpiry: Optional[int] = None
    config: Optional[Dict[str, Any]] = None


class SpawnResponse(BaseModel):
    success: bool
    alreadyActive: bool = False
    deviceId: Optional[str] = None
    brokerEndpoint: Optional[str] = None
    brokerPort: Optional[int] = None
    topic: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    sampleSchema: Optional[Dict[str, Any]] = None


@router.post("/spawn", response_model=SpawnResponse)
def spawn_device(req: SpawnRequest) -> SpawnResponse:
    # Enforce 2 instances per trial account (use profileKey as a simple trial key here)
    trial_key = req.profileKey
    count = _STATE["trial_instances"].get(trial_key, 0)
    # If device for this profile+mode already exists, return alreadyActive
    for dev_id, dev in _STATE["devices"].items():
        if dev["profile_key"] == req.profileKey and dev["mode"] == req.mode:
            return SpawnResponse(success=True, alreadyActive=True, deviceId=dev_id)

    if count >= MAX_TRIAL_INSTANCES:
        raise HTTPException(
            status_code=400,
            detail="Trial limit reached: only 2 agent instances allowed",
        )

    device_id = _random_key("dev")
    topic = f"mi/{device_id}/telemetry"
    broker_endpoint = "mqtt.xx.com"
    broker_port = 5002
    username = device_id
    password = req.profileKey

    _STATE["devices"][device_id] = {
        "profile_key": req.profileKey,
        "mode": req.mode,
        "status": "active",
        "created_at": _now_ms(),
        "topic": topic,
        "username": username,
        "password": password,
        "broker": {
            "endpoint": broker_endpoint,
            "port": broker_port,
        },
        "config": req.config or {},
    }
    _STATE["trial_instances"][trial_key] = count + 1

    return SpawnResponse(
        success=True,
        alreadyActive=False,
        deviceId=device_id,
        brokerEndpoint=broker_endpoint,
        brokerPort=broker_port,
        topic=topic,
        username=username,
        password=password,
        sampleSchema={
            "temperature": "float",
            "pressure": "float",
            "vibration": "float",
        },
    )


class HealthResponse(BaseModel):
    status: str


@router.get("/container/{device_id}/health", response_model=HealthResponse)
def device_health(device_id: str) -> HealthResponse:
    dev = _STATE["devices"].get(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")
    return HealthResponse(status=dev.get("status", "active"))


class Profile(BaseModel):
    key: str
    name: str


class ProfilesResponse(BaseModel):
    profiles: List[Profile]


@router.get("/profiles", response_model=ProfilesResponse)
def list_profiles() -> ProfilesResponse:
    profiles = [
        Profile(key=k, name=v.get("name", k)) for k, v in _STATE["profiles"].items()
    ]
    # Add some defaults for demo if empty
    if not profiles:
        defaults = [
            ("profile_demo_1", "Demo Profile 1"),
            ("profile_demo_2", "Demo Profile 2"),
        ]
        for key, name in defaults:
            _STATE["profiles"].setdefault(
                key, {"name": name, "owner_email": "demo@micro.ai"}
            )
        profiles = [Profile(key=key, name=name) for key, name in defaults]
    return ProfilesResponse(profiles=profiles)


class CreateProfileRequest(BaseModel):
    accessToken: Optional[str] = None
    profileName: str
    trainingSeconds: int
    daysToMaintenance: int
    cycleDuration: int


class CreateProfileResponse(BaseModel):
    success: bool
    profileKey: str


@router.post("/profiles", response_model=CreateProfileResponse)
def create_profile(req: CreateProfileRequest) -> CreateProfileResponse:
    profile_key = _random_key("profile")
    _STATE["profiles"][profile_key] = {
        "name": req.profileName,
        "owner_email": "demo@micro.ai",
        "config": {
            "trainingSeconds": req.trainingSeconds,
            "daysToMaintenance": req.daysToMaintenance,
            "cycleDuration": req.cycleDuration,
        },
    }
    return CreateProfileResponse(success=True, profileKey=profile_key)


class CreatePasswordRequest(BaseModel):
    email: str
    password: str
    profileKey: Optional[str] = None


class CreatePasswordResponse(BaseModel):
    success: bool
    message: str
    accessToken: str


@router.post("/create-password", response_model=CreatePasswordResponse)
def create_password(req: CreatePasswordRequest) -> CreatePasswordResponse:
    """Convert OTP-based trial account to password-based full account"""
    email = req.email.strip().lower()

    # Update account record
    record = _STATE["accounts"].get(email)
    if not record:
        _STATE["accounts"][email] = {}
        record = _STATE["accounts"][email]

    # Store password (in real system, this would be hashed)
    record["password"] = req.password
    record["is_premature"] = False
    record["has_password"] = True

    # Generate access token
    access_token = _random_key("token", 32)
    record["access_token"] = access_token
    # persist accounts
    _save_json_db(_ACCOUNTS_DB_FILE, _STATE["accounts"])

    return CreatePasswordResponse(
        success=True,
        message="Password created successfully. You are now logged in.",
        accessToken=access_token,
    )


class SubscribeRequest(BaseModel):
    userId: int
    accessToken: Optional[str] = None
    email: str
    deviceId: Optional[str] = None


class SubscribeResponse(BaseModel):
    success: bool


@router.post("/subscribe", response_model=SubscribeResponse)
def subscribe_notifications(_req: SubscribeRequest) -> SubscribeResponse:
    return SubscribeResponse(success=True)


class TicketRequest(BaseModel):
    deviceId: str
    summary: str


class TicketResponse(BaseModel):
    ticketNumber: str
    link: str


@router.post("/tickets", response_model=TicketResponse)
def create_ticket(req: TicketRequest) -> TicketResponse:
    num = str(random.randint(1000, 9999))
    return TicketResponse(
        ticketNumber=num, link=f"https://tickets.example.com/{req.deviceId}/{num}"
    )


class TransferChatRequest(BaseModel):
    anonUserId: str
    chatId: int
    toUserId: int


class TransferChatResponse(BaseModel):
    success: bool
    newSessionId: str


@router.post("/transfer-chat", response_model=TransferChatResponse)
def transfer_chat(_req: TransferChatRequest) -> TransferChatResponse:
    return TransferChatResponse(success=True, newSessionId=_random_key("session"))


# -- Users (invitation) --
class InviteUserRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    phone: Optional[str] = None
    access: Optional[list[str]] = None


class InviteUserResponse(BaseModel):
    success: bool
    userId: str


@router.post("/users", response_model=InviteUserResponse)
def invite_user(req: InviteUserRequest) -> InviteUserResponse:
    user_id = _random_key("user")
    return InviteUserResponse(success=True, userId=user_id)


# -- Auth APIs --
class LoginRequest(BaseModel):
    email: str
    password: str
    sessionId: Optional[str] = None


class LoginResponse(BaseModel):
    success: bool
    loggedIn: bool
    sessionId: str
    email: str


@router.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    email = req.email.strip().lower()
    password = req.password
    # Default admin credentials
    if email == "admin@email.com" and password:
        expected = "password"
        if password != expected:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    else:
        # check created accounts
        record = _STATE["accounts"].get(email)
        if not record or record.get("password") != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    session_id = req.sessionId or _random_key("session")
    _STATE["sessions"][session_id] = {"email": email, "created_at": _now_ms()}
    _save_json_db(_SESSIONS_DB_FILE, _STATE["sessions"])
    return LoginResponse(success=True, loggedIn=True, sessionId=session_id, email=email)


class LogoutRequest(BaseModel):
    sessionId: str


class LogoutResponse(BaseModel):
    success: bool
    loggedOut: bool


@router.post("/auth/logout", response_model=LogoutResponse)
def logout(req: LogoutRequest) -> LogoutResponse:
    _STATE["sessions"].pop(req.sessionId, None)
    _save_json_db(_SESSIONS_DB_FILE, _STATE["sessions"])
    return LogoutResponse(success=True, loggedOut=True)


class SessionStatusRequest(BaseModel):
    sessionId: str


class SessionStatusResponse(BaseModel):
    success: bool
    loggedIn: bool
    email: Optional[str] = None


@router.post("/auth/status", response_model=SessionStatusResponse)
def session_status(req: SessionStatusRequest) -> SessionStatusResponse:
    sess = _STATE["sessions"].get(req.sessionId)
    if not sess:
        return SessionStatusResponse(success=True, loggedIn=False, email=None)
    return SessionStatusResponse(success=True, loggedIn=True, email=sess.get("email"))


# -- Validation APIs --
class ValidateBrokerRequest(BaseModel):
    profileKey: str
    brokerEndpoint: Optional[str] = "mqtt.micro.ai"
    brokerPort: Optional[int] = 5002


class ValidateBrokerResponse(BaseModel):
    success: bool
    canConnect: bool
    message: str


@router.post("/validate-broker", response_model=ValidateBrokerResponse)
def validate_broker_connection(req: ValidateBrokerRequest) -> ValidateBrokerResponse:
    """Mock broker connection validation - always succeeds for demo"""
    return ValidateBrokerResponse(
        success=True,
        canConnect=True,
        message="Broker connection validated successfully",
    )


class ValidateSchemaRequest(BaseModel):
    profileKey: str
    schema: Optional[Dict[str, Any]] = None


class ValidateSchemaResponse(BaseModel):
    success: bool
    isValid: bool
    message: str
    suggestedSchema: Optional[Dict[str, Any]] = None


@router.post("/validate-schema", response_model=ValidateSchemaResponse)
def validate_schema(req: ValidateSchemaRequest) -> ValidateSchemaResponse:
    """Mock schema validation - always succeeds for demo"""
    suggested = {
        "temperature": "float",
        "pressure": "float",
        "vibration": "float",
        "speed": "float",
        "gyro_x": "float",
        "gyro_y": "float",
        "gyro_z": "float",
    }
    return ValidateSchemaResponse(
        success=True, isValid=True, message="Schema is valid", suggestedSchema=suggested
    )


# -- Payment APIs --
class PaymentPlan(BaseModel):
    id: str
    name: str
    price: float
    devicesLimit: int


class ListPaymentPlansResponse(BaseModel):
    plans: List[PaymentPlan]


@router.get("/payment-plans", response_model=ListPaymentPlansResponse)
def list_payment_plans() -> ListPaymentPlansResponse:
    """Mock payment plans"""
    return ListPaymentPlansResponse(
        plans=[
            PaymentPlan(id="plan_trial", name="Trial", price=0.0, devicesLimit=2),
            PaymentPlan(id="plan_basic", name="Basic", price=99.0, devicesLimit=10),
            PaymentPlan(
                id="plan_pro", name="Professional", price=299.0, devicesLimit=50
            ),
            PaymentPlan(
                id="plan_enterprise", name="Enterprise", price=999.0, devicesLimit=999
            ),
        ]
    )


class ProcessPaymentRequest(BaseModel):
    planId: str
    cardNumber: str
    expiryMonth: int
    expiryYear: int
    cvv: str
    billingEmail: str


class ProcessPaymentResponse(BaseModel):
    success: bool
    transactionId: str
    message: str


@router.post("/process-payment", response_model=ProcessPaymentResponse)
def process_payment(req: ProcessPaymentRequest) -> ProcessPaymentResponse:
    """Mock payment processing - always succeeds for demo"""
    transaction_id = _random_key("txn")
    return ProcessPaymentResponse(
        success=True,
        transactionId=transaction_id,
        message="Payment processed successfully",
    )


# -- Dashboard APIs --
class DashboardViewRequest(BaseModel):
    deviceId: str
    assetId: Optional[str] = None


class DashboardViewResponse(BaseModel):
    success: bool
    dashboardUrl: str
    widgets: List[Dict[str, Any]]


@router.post("/dashboard/view", response_model=DashboardViewResponse)
def get_dashboard_view(req: DashboardViewRequest) -> DashboardViewResponse:
    """Mock dashboard view API"""
    return DashboardViewResponse(
        success=True,
        dashboardUrl=f"https://dashboard.micro.ai/device/{req.deviceId}",
        widgets=[
            {"type": "health-score", "data": {"score": 87.5}},
            {"type": "vibration-chart", "data": {"channel": "gyro_x"}},
            {"type": "temperature-chart", "data": {"channel": "temperature"}},
        ],
    )


class SwitchChannelRequest(BaseModel):
    deviceId: str
    widgetId: str
    fromChannel: str
    toChannel: str


class SwitchChannelResponse(BaseModel):
    success: bool
    message: str


@router.post("/dashboard/switch-channel", response_model=SwitchChannelResponse)
def switch_dashboard_channel(req: SwitchChannelRequest) -> SwitchChannelResponse:
    """Mock channel switching API"""
    return SwitchChannelResponse(
        success=True,
        message=f"Switched widget from {req.fromChannel} to {req.toChannel}",
    )


class RefreshDashboardRequest(BaseModel):
    deviceId: str


class RefreshDashboardResponse(BaseModel):
    success: bool
    message: str


@router.post("/dashboard/refresh", response_model=RefreshDashboardResponse)
def refresh_dashboard(req: RefreshDashboardRequest) -> RefreshDashboardResponse:
    """Mock dashboard refresh API"""
    return RefreshDashboardResponse(
        success=True, message="Dashboard refreshed successfully"
    )


# -- Metrics & Analytics APIs --
class QueryMetricsRequest(BaseModel):
    deviceId: str
    metrics: List[str]
    startTime: str
    endTime: str


class QueryMetricsResponse(BaseModel):
    success: bool
    data: Dict[str, List[Dict[str, Any]]]


@router.post("/metrics/query", response_model=QueryMetricsResponse)
def query_metrics(req: QueryMetricsRequest) -> QueryMetricsResponse:
    """Mock time-series metrics query"""
    # Generate sample data for requested metrics
    data = {}
    for metric in req.metrics:
        data[metric] = [
            {"timestamp": "2025-10-14T10:00:00Z", "value": random.uniform(20, 100)},
            {"timestamp": "2025-10-14T11:00:00Z", "value": random.uniform(20, 100)},
            {"timestamp": "2025-10-14T12:00:00Z", "value": random.uniform(20, 100)},
        ]
    return QueryMetricsResponse(success=True, data=data)


class CorrelationRequest(BaseModel):
    deviceId: str
    metric1: str
    metric2: str
    startTime: str
    endTime: str


class CorrelationResponse(BaseModel):
    success: bool
    correlation: float
    samples: List[Dict[str, float]]


@router.post("/metrics/correlation", response_model=CorrelationResponse)
def compute_correlation(req: CorrelationRequest) -> CorrelationResponse:
    """Mock correlation computation"""
    correlation = random.uniform(0.3, 0.9)
    samples = [
        {req.metric1: random.uniform(20, 80), req.metric2: random.uniform(20, 80)}
        for _ in range(10)
    ]
    return CorrelationResponse(
        success=True, correlation=round(correlation, 2), samples=samples
    )


# -- Alerts APIs --
class CreateAlertRequest(BaseModel):
    deviceId: str
    alertType: str  # "threshold" or "predictive"
    metric: Optional[str] = None
    threshold: Optional[float] = None
    duration: Optional[int] = None  # seconds
    params: Optional[Dict[str, Any]] = None


class CreateAlertResponse(BaseModel):
    success: bool
    alertId: str
    message: str


@router.post("/alerts/create", response_model=CreateAlertResponse)
def create_alert(req: CreateAlertRequest) -> CreateAlertResponse:
    """Mock alert creation"""
    alert_id = _random_key("alert")
    return CreateAlertResponse(
        success=True,
        alertId=alert_id,
        message=f"Alert created successfully for {req.alertType}",
    )


# -- Advanced Ticketing APIs --
class GetAnomalyRequest(BaseModel):
    deviceId: str
    metric: str
    timestamp: str


class GetAnomalyResponse(BaseModel):
    success: bool
    eventId: str
    details: Dict[str, Any]


@router.post("/anomalies/get", response_model=GetAnomalyResponse)
def get_anomaly(req: GetAnomalyRequest) -> GetAnomalyResponse:
    """Mock anomaly retrieval"""
    event_id = _random_key("event")
    return GetAnomalyResponse(
        success=True,
        eventId=event_id,
        details={
            "metric": req.metric,
            "timestamp": req.timestamp,
            "severity": "high",
            "value": random.uniform(80, 120),
            "expectedRange": [20, 70],
        },
    )


class AttachToTicketRequest(BaseModel):
    ticketNumber: str
    attachmentType: str
    attachmentData: Dict[str, Any]


class AttachToTicketResponse(BaseModel):
    success: bool
    message: str


@router.post("/tickets/attach", response_model=AttachToTicketResponse)
def attach_to_ticket(req: AttachToTicketRequest) -> AttachToTicketResponse:
    """Mock ticket attachment"""
    return AttachToTicketResponse(
        success=True,
        message=f"Attached {req.attachmentType} to ticket {req.ticketNumber}",
    )


class LookupUserRequest(BaseModel):
    query: str  # name or email


class LookupUserResponse(BaseModel):
    success: bool
    userId: Optional[str] = None
    userName: Optional[str] = None
    userEmail: Optional[str] = None


@router.post("/users/lookup", response_model=LookupUserResponse)
def lookup_user(req: LookupUserRequest) -> LookupUserResponse:
    """Mock user lookup"""
    user_id = _random_key("user")
    return LookupUserResponse(
        success=True,
        userId=user_id,
        userName=req.query,
        userEmail=f"{req.query.lower().replace(' ', '.')}@example.com",
    )


class UpdateTicketRequest(BaseModel):
    ticketNumber: str
    assigneeId: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None


class UpdateTicketResponse(BaseModel):
    success: bool
    message: str


@router.post("/tickets/update", response_model=UpdateTicketResponse)
def update_ticket(req: UpdateTicketRequest) -> UpdateTicketResponse:
    """Mock ticket update"""
    return UpdateTicketResponse(
        success=True, message=f"Ticket {req.ticketNumber} updated successfully"
    )


# -- Predictions & Explanations APIs --
class ForecastMaintenanceRequest(BaseModel):
    deviceId: str
    horizonDays: int = 30


class ForecastMaintenanceResponse(BaseModel):
    success: bool
    predictedDate: str
    daysUntilMaintenance: int
    confidence: float


@router.post("/predictions/maintenance", response_model=ForecastMaintenanceResponse)
def forecast_maintenance(
    req: ForecastMaintenanceRequest,
) -> ForecastMaintenanceResponse:
    """Mock maintenance prediction"""
    days_until = random.randint(5, 25)
    from datetime import datetime, timedelta

    predicted_date = (datetime.now() + timedelta(days=days_until)).strftime("%Y-%m-%d")
    return ForecastMaintenanceResponse(
        success=True,
        predictedDate=predicted_date,
        daysUntilMaintenance=days_until,
        confidence=0.87,
    )


class CreateScheduledTicketRequest(BaseModel):
    deviceId: str
    scheduledDate: str
    summary: str


class CreateScheduledTicketResponse(BaseModel):
    success: bool
    ticketNumber: str
    scheduledFor: str


@router.post("/tickets/schedule", response_model=CreateScheduledTicketResponse)
def create_scheduled_ticket(
    req: CreateScheduledTicketRequest,
) -> CreateScheduledTicketResponse:
    """Mock scheduled ticket creation"""
    ticket_num = str(random.randint(1000, 9999))
    return CreateScheduledTicketResponse(
        success=True, ticketNumber=ticket_num, scheduledFor=req.scheduledDate
    )


class ExplainHealthScoreRequest(BaseModel):
    deviceId: str
    date: Optional[str] = None


class ExplainHealthScoreResponse(BaseModel):
    success: bool
    healthScore: float
    topFactors: List[Dict[str, Any]]


@router.post("/predictions/explain-health", response_model=ExplainHealthScoreResponse)
def explain_health_score(req: ExplainHealthScoreRequest) -> ExplainHealthScoreResponse:
    """Mock health score explanation"""
    return ExplainHealthScoreResponse(
        success=True,
        healthScore=87.5,
        topFactors=[
            {"factor": "Elevated vibration", "impact": -8.5, "metric": "vibration"},
            {"factor": "Longer cycle duration", "impact": -4.0, "metric": "cycle_time"},
            {"factor": "Temperature stable", "impact": +2.0, "metric": "temperature"},
        ],
    )


class CompareDevicesRequest(BaseModel):
    deviceIds: List[str]
    metric: str
    startTime: str
    endTime: str


class CompareDevicesResponse(BaseModel):
    success: bool
    comparison: Dict[str, Dict[str, Any]]
    summary: str


@router.post("/devices/compare", response_model=CompareDevicesResponse)
def compare_devices(req: CompareDevicesRequest) -> CompareDevicesResponse:
    """Mock cross-device comparison"""
    comparison = {}
    for device_id in req.deviceIds:
        comparison[device_id] = {
            "average": random.uniform(40, 80),
            "variance": random.uniform(2, 15),
            "trend": random.choice(["increasing", "stable", "decreasing"]),
        }
    return CompareDevicesResponse(
        success=True,
        comparison=comparison,
        summary=f"Compared {len(req.deviceIds)} devices for metric {req.metric}",
    )
