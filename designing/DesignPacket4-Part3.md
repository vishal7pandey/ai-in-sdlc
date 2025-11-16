# Design Packet 4: Part 3 - Security, Reliability & Production Checklist

## 4. Security, Privacy & Compliance

### 4.1 Application Security

#### 4.1.1 JWT Hardening

```python
# src/auth/jwt.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Store in env/secrets manager
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token blacklist (use Redis in production)
revoked_tokens = set()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": secrets.token_urlsafe(16),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, expected_type: str = "access"):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check token type
        if payload.get("type") != expected_type:
            raise JWTError("Invalid token type")

        # Check if revoked
        jti = payload.get("jti")
        if jti in revoked_tokens:
            raise JWTError("Token has been revoked")

        return payload

    except JWTError:
        return None

def revoke_token(token: str):
    \"\"\"Add token to blacklist\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        jti = payload.get("jti")
        if jti:
            revoked_tokens.add(jti)
            # In production, store in Redis with TTL = token expiry
    except JWTError:
        pass

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

#### 4.1.2 Input Sanitization & Validation

```python
# src/utils/sanitization.py
from bleach import clean
from typing import Any, Dict
import re

# Allowed HTML tags (for Markdown support in chat)
ALLOWED_TAGS = ['p', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'a']
ALLOWED_ATTRIBUTES = {'a': ['href', 'title']}

def sanitize_html(text: str) -> str:
    \"\"\"Remove potentially dangerous HTML\"\"\"
    return clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, strip=True)

def sanitize_sql_injection(text: str) -> str:
    \"\"\"Prevent SQL injection (even though we use ORM)\"\"\"
    # Remove SQL keywords from user input
    dangerous_patterns = [
        r"(\bDROP\b|\bDELETE\b|\bUPDATE\b|\bINSERT\b|\bEXEC\b|\bEXECUTE\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bUNION\b.*\bSELECT\b)"
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text

def sanitize_prompt_injection(text: str) -> str:
    \"\"\"Detect and neutralize prompt injection attempts\"\"\"

    # Patterns that indicate prompt injection
    injection_patterns = [
        r"ignore (previous|all) instructions?",
        r"you are now",
        r"system: ",
        r"</s>",
        r"<\|endoftext\|>",
        r"disregard",
        r"forget (everything|all)",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Flag for review or block
            raise ValueError(f"Potential prompt injection detected: {pattern}")

    return text

def validate_requirement_input(req_data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Validate and sanitize requirement input\"\"\"

    # Sanitize text fields
    if "title" in req_data:
        req_data["title"] = sanitize_html(req_data["title"])[:500]  # Max length

    if "action" in req_data:
        req_data["action"] = sanitize_html(req_data["action"])[:2000]

    if "rationale" in req_data:
        req_data["rationale"] = sanitize_html(req_data["rationale"])[:5000]

    # Validate enums
    valid_types = ["functional", "non-functional", "business", "security", "data", "interface", "constraint"]
    if req_data.get("type") not in valid_types:
        raise ValueError(f"Invalid requirement type: {req_data.get('type')}")

    valid_priorities = ["low", "medium", "high", "must"]
    if req_data.get("priority") not in valid_priorities:
        raise ValueError(f"Invalid priority: {req_data.get('priority')}")

    # Validate confidence range
    if "confidence" in req_data:
        confidence = float(req_data["confidence"])
        if not (0 <= confidence <= 1):
            raise ValueError(f"Confidence must be between 0 and 1: {confidence}")

    return req_data
```

#### 4.1.3 API Rate Limiting

```python
# src/middleware/rate_limit.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Initialize Redis for distributed rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)

# Rate limit tiers
RATE_LIMITS = {
    "anonymous": "10/minute",
    "authenticated": "100/minute",
    "premium": "1000/minute"
}

def get_user_tier(request: Request) -> str:
    \"\"\"Determine user's rate limit tier\"\"\"
    # Check if authenticated
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return "anonymous"

    # Extract user from token
    # ... (JWT verification logic)

    # Check user's subscription tier
    # user = get_user_from_token(auth_header)
    # return user.tier

    return "authenticated"

async def rate_limit_middleware(request: Request, call_next):
    \"\"\"Custom rate limiting middleware\"\"\"

    # Determine tier
    tier = get_user_tier(request)

    # Get rate limit for tier
    limit = RATE_LIMITS[tier]

    # Check rate limit
    key = f"rate_limit:{tier}:{get_remote_address(request)}"

    # Increment counter
    current = redis_client.incr(key)

    if current == 1:
        # First request, set expiry
        redis_client.expire(key, 60)  # 1 minute window

    # Parse limit (e.g., "100/minute")
    max_requests = int(limit.split("/")[0])

    if current > max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Tier: {tier}, Limit: {limit}"
        )

    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(max_requests)
    response.headers["X-RateLimit-Remaining"] = str(max(0, max_requests - current))
    response.headers["X-RateLimit-Reset"] = str(redis_client.ttl(key))

    return response
```

#### 4.1.4 WebSocket Message Validation

```python
# src/websocket/validation.py
from pydantic import BaseModel, validator, ValidationError
from typing import Literal, Optional, Dict, Any

class WebSocketMessage(BaseModel):
    type: Literal[
        "ping", "chat_message", "requirement_update",
        "accept_inferred", "request_state"
    ]
    payload: Dict[str, Any]

    @validator("payload")
    def validate_payload_size(cls, v):
        # Prevent DoS via large payloads
        import sys
        size = sys.getsizeof(str(v))
        if size > 1_000_000:  # 1MB limit
            raise ValueError("Payload too large")
        return v

class ChatMessagePayload(BaseModel):
    session_id: str
    message: str
    timestamp: str

    @validator("message")
    def validate_message_length(cls, v):
        if len(v) > 10_000:  # 10k char limit
            raise ValueError("Message too long")
        return v

class RequirementUpdatePayload(BaseModel):
    requirement_id: str
    updates: Dict[str, Any]

    @validator("requirement_id")
    def validate_req_id_format(cls, v):
        import re
        if not re.match(r"^REQ(-INF)?-\d{3,}$", v):
            raise ValueError("Invalid requirement ID format")
        return v

async def validate_websocket_message(data: dict) -> WebSocketMessage:
    \"\"\"Validate incoming WebSocket message\"\"\"
    try:
        # Validate base message structure
        message = WebSocketMessage(**data)

        # Validate payload based on type
        if message.type == "chat_message":
            ChatMessagePayload(**message.payload)
        elif message.type == "requirement_update":
            RequirementUpdatePayload(**message.payload)

        return message

    except ValidationError as e:
        raise ValueError(f"Invalid message format: {e}")
```

### 4.2 LLM-Specific Security

#### 4.2.1 Prompt Injection Defenses

```python
# src/agents/security/prompt_guard.py
import re
from typing import Tuple

class PromptGuard:
    \"\"\"Guard against prompt injection attacks\"\"\"

    # Known injection patterns
    INJECTION_PATTERNS = [
        # Instruction override attempts
        (r"ignore (previous|all|above) instructions?", "high"),
        (r"disregard (previous|all) (instructions?|prompts?)", "high"),
        (r"forget (everything|all|previous)", "high"),

        # Role manipulation
        (r"you are now (a|an)?", "high"),
        (r"act as (a|an)?", "medium"),
        (r"pretend (you are|to be)", "medium"),

        # System prompt leakage attempts
        (r"(show|reveal|tell me) (your|the) (system )?(prompt|instructions)", "high"),
        (r"what (are|were) you told", "medium"),

        # Special tokens
        (r"<\|endoftext\|>", "high"),
        (r"</s>", "high"),
        (r"<\|im_start\|>", "high"),

        # Encoding tricks
        (r"base64|hex encode|rot13", "medium"),
    ]

    def scan(self, text: str) -> Tuple[bool, str, list]:
        \"\"\"
        Scan text for prompt injection attempts

        Returns:
            (is_safe, risk_level, detected_patterns)
        \"\"\"
        detected = []
        max_severity = "low"

        for pattern, severity in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append({
                    "pattern": pattern,
                    "severity": severity
                })

                # Track highest severity
                if severity == "high":
                    max_severity = "high"
                elif severity == "medium" and max_severity != "high":
                    max_severity = "medium"

        is_safe = max_severity != "high"

        return is_safe, max_severity, detected

    def sanitize(self, text: str) -> str:
        \"\"\"Remove or neutralize injection attempts\"\"\"

        # Replace special tokens
        text = re.sub(r"<\|endoftext\|>", "[REMOVED]", text, flags=re.IGNORECASE)
        text = re.sub(r"</s>", "[REMOVED]", text, flags=re.IGNORECASE)

        # Remove instruction override attempts
        text = re.sub(
            r"(ignore|disregard|forget) (previous|all|above) (instructions?|prompts?)",
            "[REMOVED]",
            text,
            flags=re.IGNORECASE
        )

        return text

# Usage in agent
prompt_guard = PromptGuard()

async def extraction_agent_with_guard(state: GraphState) -> GraphState:
    # Get user message
    user_message = state["messages"][-1].content

    # Scan for injection
    is_safe, risk_level, detected = prompt_guard.scan(user_message)

    if not is_safe:
        # Log security event
        log_security_event(
            event_type="prompt_injection_detected",
            session_id=state["session_id"],
            user_id=state["user_id"],
            detected_patterns=detected,
            risk_level=risk_level
        )

        # Reject or sanitize
        if risk_level == "high":
            raise ValueError("Message rejected due to security concerns")
        else:
            # Sanitize and proceed with warning
            user_message = prompt_guard.sanitize(user_message)

    # Continue with agent logic
    # ...
```

#### 4.2.2 Sensitive Data Filtering

```python
# src/agents/security/pii_filter.py
import re
from typing import List, Dict

class PIIFilter:
    \"\"\"Filter personally identifiable information\"\"\"

    # Regex patterns for common PII
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def detect(self, text: str) -> Dict[str, List[str]]:
        \"\"\"Detect PII in text\"\"\"
        findings = {}

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches

        return findings

    def redact(self, text: str) -> str:
        \"\"\"Redact PII from text\"\"\"
        for pii_type, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)

        return text

# Usage in validation agent
pii_filter = PIIFilter()

async def validation_agent_with_pii_check(state: GraphState) -> GraphState:
    # Check requirements for PII
    for req in state["requirements"]:
        # Check all text fields
        for field in ["title", "action", "condition", "rationale"]:
            text = getattr(req, field, "")
            if text:
                pii_detected = pii_filter.detect(text)

                if pii_detected:
                    # Log finding
                    log_security_event(
                        event_type="pii_detected",
                        requirement_id=req.id,
                        pii_types=list(pii_detected.keys())
                    )

                    # Add validation issue
                    state["validation_issues"].append({
                        "requirement_id": req.id,
                        "severity": "warning",
                        "message": f"PII detected: {', '.join(pii_detected.keys())}",
                        "field": field
                    })

    return state
```

### 4.3 Data Security

#### 4.3.1 Encryption at Rest

```python
# src/storage/encryption.py
from cryptography.fernet import Fernet
import base64
import os

class FieldEncryption:
    \"\"\"Encrypt sensitive fields in database\"\"\"

    def __init__(self):
        # Load encryption key from environment or secrets manager
        key = os.getenv("FIELD_ENCRYPTION_KEY")
        if not key:
            # Generate new key (store securely!)
            key = Fernet.generate_key()

        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> str:
        \"\"\"Encrypt string data\"\"\"
        if not data:
            return data

        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        \"\"\"Decrypt string data\"\"\"
        if not encrypted_data:
            return encrypted_data

        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()

# SQLAlchemy hybrid property for transparent encryption
from sqlalchemy import String, TypeDecorator

class EncryptedString(TypeDecorator):
    impl = String
    cache_ok = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encryptor = FieldEncryption()

    def process_bind_param(self, value, dialect):
        if value is not None:
            return self.encryptor.encrypt(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return self.encryptor.decrypt(value)
        return value

# Usage in model
from sqlalchemy import Column
from src.models.database import Base

class SensitiveData(Base):
    __tablename__ = "sensitive_data"

    id = Column(String, primary_key=True)
    # This field will be automatically encrypted/decrypted
    confidential_info = Column(EncryptedString(500))
```

#### 4.3.2 Tamper-Proof RD Versions

```python
# src/storage/integrity.py
import hashlib
import hmac
import json
from typing import Dict, Any

class DocumentIntegrity:
    \"\"\"Ensure RD documents haven't been tampered with\"\"\"

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def compute_hash(self, document: Dict[str, Any]) -> str:
        \"\"\"Compute HMAC of document for integrity verification\"\"\"

        # Create canonical representation
        canonical = json.dumps(document, sort_keys=True)

        # Compute HMAC-SHA256
        signature = hmac.new(
            self.secret_key,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_integrity(self, document: Dict[str, Any], signature: str) -> bool:
        \"\"\"Verify document hasn't been tampered with\"\"\"
        computed = self.compute_hash(document)
        return hmac.compare_digest(computed, signature)

    def sign_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Add integrity signature to document\"\"\"
        document_copy = document.copy()

        # Remove existing signature if present
        document_copy.pop("_signature", None)

        # Compute signature
        signature = self.compute_hash(document_copy)

        # Add signature
        document_copy["_signature"] = signature
        document_copy["_signed_at"] = datetime.utcnow().isoformat()

        return document_copy

# Usage in RD generation
integrity_service = DocumentIntegrity(secret_key=os.getenv("INTEGRITY_SECRET"))

async def generate_rd(state: GraphState) -> GraphState:
    # Generate RD content
    rd_content = synthesis_agent.generate(state)

    # Create document object
    rd_document = {
        "session_id": state["session_id"],
        "version": state["rd_version"] + 1,
        "content": rd_content,
        "requirements": [req.dict() for req in state["requirements"]],
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "user_id": state["user_id"]
        }
    }

    # Sign document
    signed_document = integrity_service.sign_document(rd_document)

    # Store with signature
    await save_rd_document(signed_document)

    return state

async def verify_rd_integrity(rd_id: str, version: int):
    \"\"\"Verify RD hasn't been tampered with\"\"\"
    document = await load_rd_document(rd_id, version)

    signature = document.pop("_signature", None)

    if not signature:
        raise ValueError("Document is not signed")

    is_valid = integrity_service.verify_integrity(document, signature)

    if not is_valid:
        # Log security event
        log_security_event(
            event_type="document_tampering_detected",
            rd_id=rd_id,
            version=version
        )
        raise ValueError("Document integrity check failed - possible tampering")

    return document
```

### 4.4 Compliance (GDPR, SOC2)

#### 4.4.1 Right to Erasure (GDPR)

```python
# src/compliance/gdpr.py
from sqlalchemy import select, delete
from src.models.database import SessionModel, RequirementModel, ChatMessageModel, AuditLogModel

class GDPRCompliance:
    \"\"\"Handle GDPR compliance operations\"\"\"

    async def erase_user_data(self, user_id: str, db_session):
        \"\"\"
        Erase all user data per GDPR Article 17

        Note: This is complex in multi-agent context due to:
        - Requirements may be co-authored
        - Chat history involves multiple users
        - RD documents may have approvals from multiple users
        \"\"\"

        # 1. Anonymize chat messages
        stmt = select(ChatMessageModel).where(ChatMessageModel.user_id == user_id)
        messages = await db_session.execute(stmt)

        for message in messages.scalars():
            message.content = "[REDACTED - User data erased]"
            message.user_id = "ANONYMIZED"

        # 2. Handle requirements
        # Option A: Delete if user is sole author
        # Option B: Anonymize user references
        stmt = select(RequirementModel).join(SessionModel).where(
            SessionModel.user_id == user_id
        )
        requirements = await db_session.execute(stmt)

        for req in requirements.scalars():
            # Check if requirement has other contributors
            # If sole author, delete
            # If co-authored, anonymize
            pass

        # 3. Delete or anonymize sessions
        stmt = select(SessionModel).where(SessionModel.user_id == user_id)
        sessions = await db_session.execute(stmt)

        for session in sessions.scalars():
            # Check if session has other participants
            # Decision: anonymize vs delete
            session.user_id = "ANONYMIZED"

        # 4. Keep audit logs but anonymize (for compliance)
        stmt = select(AuditLogModel).where(AuditLogModel.user_id == user_id)
        logs = await db_session.execute(stmt)

        for log in logs.scalars():
            log.user_id = "ANONYMIZED"
            log.ip_address = None
            log.user_agent = None

        await db_session.commit()

        # 5. Log erasure event
        log_compliance_event(
            event_type="gdpr_erasure_completed",
            user_id=user_id,
            timestamp=datetime.utcnow()
        )

    async def export_user_data(self, user_id: str, db_session) -> Dict:
        \"\"\"
        Export all user data per GDPR Article 20 (Data Portability)
        \"\"\"

        export_package = {
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "data": {}
        }

        # Sessions
        stmt = select(SessionModel).where(SessionModel.user_id == user_id)
        sessions = await db_session.execute(stmt)
        export_package["data"]["sessions"] = [
            session.to_dict() for session in sessions.scalars()
        ]

        # Chat messages
        stmt = select(ChatMessageModel).join(SessionModel).where(
            SessionModel.user_id == user_id
        )
        messages = await db_session.execute(stmt)
        export_package["data"]["messages"] = [
            msg.to_dict() for msg in messages.scalars()
        ]

        # Requirements
        # ... similar pattern

        return export_package
```

#### 4.4.2 Audit Trail Requirements

```python
# src/compliance/audit.py
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

class AuditAction(str, Enum):
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_DELETED = "session_deleted"
    REQUIREMENT_CREATED = "requirement_created"
    REQUIREMENT_UPDATED = "requirement_updated"
    REQUIREMENT_DELETED = "requirement_deleted"
    RD_GENERATED = "rd_generated"
    RD_APPROVED = "rd_approved"
    RD_EXPORTED = "rd_exported"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGED = "permission_changed"

async def create_audit_log(
    action: AuditAction,
    user_id: str,
    session_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    \"\"\"Create immutable audit log entry\"\"\"

    from src.models.database import AuditLogModel

    audit_entry = AuditLogModel(
        session_id=session_id,
        user_id=user_id,
        action=action.value,
        entity_type=entity_type,
        entity_id=entity_id,
        changes=changes,  # Before/after diff
        timestamp=datetime.utcnow(),
        ip_address=ip_address,
        user_agent=user_agent
    )

    async with get_db_session() as db:
        db.add(audit_entry)
        await db.commit()

    # Also send to external audit log service (immutable storage)
    await send_to_external_audit_service(audit_entry.to_dict())
```

---

## 5. Reliability Engineering & Failure Mode Design

### 5.1 WebSocket Reliability

#### 5.1.1 Backpressure Handling

```python
# src/websocket/backpressure.py
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

@dataclass
class QueueMetrics:
    size: int
    max_size: int
    dropped_messages: int

class BackpressureQueue:
    \"\"\"Queue with backpressure handling for WebSocket messages\"\"\"

    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size
        self.dropped_count = 0
        self.lock = asyncio.Lock()

    async def enqueue(self, message: Any, priority: int = 0) -> bool:
        \"\"\"
        Add message to queue

        Returns:
            True if queued, False if dropped
        \"\"\"
        async with self.lock:
            if len(self.queue) >= self.max_size:
                # Queue full, apply backpressure

                # Option 1: Drop oldest non-critical message
                if priority > 0:
                    # Remove oldest low-priority message
                    for i, (p, msg) in enumerate(self.queue):
                        if p == 0:
                            del self.queue[i]
                            self.dropped_count += 1
                            break
                else:
                    # Drop this message
                    self.dropped_count += 1
                    return False

            self.queue.append((priority, message))
            return True

    async def dequeue(self) -> Any:
        \"\"\"Remove and return next message\"\"\"
        async with self.lock:
            if self.queue:
                priority, message = self.queue.popleft()
                return message
            return None

    def metrics(self) -> QueueMetrics:
        return QueueMetrics(
            size=len(self.queue),
            max_size=self.max_size,
            dropped_messages=self.dropped_count
        )

# Usage in WebSocket manager
class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.message_queues: Dict[str, BackpressureQueue] = {}

    async def send_message(self, session_id: str, message: dict, priority: int = 0):
        \"\"\"Send message with backpressure handling\"\"\"

        queue = self.message_queues.get(session_id)
        if not queue:
            queue = BackpressureQueue()
            self.message_queues[session_id] = queue

        # Try to queue
        queued = await queue.enqueue(message, priority)

        if not queued:
            # Log dropped message
            log_warning(
                "websocket_message_dropped",
                session_id=session_id,
                queue_size=queue.metrics().size
            )

        # Try to flush queue
        await self.flush_queue(session_id)

    async def flush_queue(self, session_id: str):
        \"\"\"Flush message queue to WebSocket\"\"\"

        websocket = self.connections.get(session_id)
        queue = self.message_queues.get(session_id)

        if not websocket or not queue:
            return

        while True:
            message = await queue.dequeue()
            if not message:
                break

            try:
                await websocket.send_json(message)
            except Exception as e:
                # Re-queue on error
                await queue.enqueue(message, priority=1)
                break
```

#### 5.1.2 Ordered Delivery Guarantee

```python
# src/websocket/ordered_delivery.py
from typing import Dict, List
import asyncio

class OrderedMessageQueue:
    \"\"\"Ensure messages are delivered in order\"\"\"

    def __init__(self):
        self.sequence_number = 0
        self.pending_messages: Dict[int, dict] = {}
        self.next_expected = 0
        self.lock = asyncio.Lock()

    async def add_message(self, message: dict) -> List[dict]:
        \"\"\"
        Add message and return any messages ready to send in order

        Returns:
            List of messages that can now be sent in order
        \"\"\"
        async with self.lock:
            # Assign sequence number
            seq = self.sequence_number
            self.sequence_number += 1

            message["_seq"] = seq
            self.pending_messages[seq] = message

            # Check if we can send any messages
            ready = []
            while self.next_expected in self.pending_messages:
                ready.append(self.pending_messages.pop(self.next_expected))
                self.next_expected += 1

            return ready

    async def wait_for_sequence(self, seq: int, timeout: float = 5.0):
        \"\"\"Wait until a specific sequence number is ready\"\"\"
        start_time = asyncio.get_event_loop().time()

        while True:
            async with self.lock:
                if seq in self.pending_messages:
                    return self.pending_messages[seq]

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Sequence {seq} not received within timeout")

            await asyncio.sleep(0.01)
```

### 5.2 LangGraph Resilience

#### 5.2.1 Node Retry Logic

```python
# src/orchestrator/resilience.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

# Retry decorator for agent nodes
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def resilient_agent_node(agent_func, state: GraphState):
    \"\"\"
    Wrapper for agent nodes with retry logic

    Retries on:
    - Network errors
    - Timeout errors
    - Transient LLM failures

    Does NOT retry on:
    - Validation errors
    - Business logic errors
    - Permanent failures
    \"\"\"
    try:
        return await agent_func(state)
    except Exception as e:
        # Log attempt
        logger.warning(
            f"Agent node failed (will retry): {type(e).__name__}: {e}",
            extra={
                "agent": agent_func.__name__,
                "session_id": state.get("session_id"),
                "attempt": getattr(e, '__context__', {}).get('attempt', 1)
            }
        )
        raise

# Circuit breaker pattern
class CircuitBreaker:
    \"\"\"Circuit breaker for LLM calls\"\"\"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func, *args, **kwargs):
        \"\"\"Execute function with circuit breaker\"\"\"

        if self.state == "open":
            # Check if we should try recovery
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success - reset circuit
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise

class CircuitBreakerOpenError(Exception):
    pass
```

#### 5.2.2 Checkpoint Corruption Recovery

```python
# src/orchestrator/checkpoint_recovery.py
from langgraph.checkpoint.base import BaseCheckpointSaver
import json
from typing import Optional

class ResilientCheckpointSaver(BaseCheckpointSaver):
    \"\"\"Checkpoint saver with corruption recovery\"\"\"

    async def get_checkpoint(self, thread_id: str, checkpoint_id: str):
        \"\"\"Retrieve checkpoint with validation\"\"\"
        try:
            checkpoint = await self._load_checkpoint(thread_id, checkpoint_id)

            # Validate checkpoint integrity
            if not self._validate_checkpoint(checkpoint):
                logger.warning(
                    f"Checkpoint {checkpoint_id} failed validation, attempting recovery"
                )

                # Try to recover from previous checkpoint
                checkpoint = await self._recover_checkpoint(thread_id, checkpoint_id)

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

            # Fallback to last known good checkpoint
            return await self._get_last_known_good(thread_id)

    def _validate_checkpoint(self, checkpoint: dict) -> bool:
        \"\"\"Validate checkpoint structure and data\"\"\"
        required_fields = ["session_id", "messages", "requirements"]

        for field in required_fields:
            if field not in checkpoint:
                return False

        # Validate data types
        if not isinstance(checkpoint["messages"], list):
            return False

        return True

    async def _recover_checkpoint(self, thread_id: str, checkpoint_id: str):
        \"\"\"Attempt to recover corrupted checkpoint\"\"\"

        # Strategy 1: Load from previous checkpoint
        parent_id = await self._get_parent_checkpoint_id(thread_id, checkpoint_id)
        if parent_id:
            parent = await self._load_checkpoint(thread_id, parent_id)
            if self._validate_checkpoint(parent):
                logger.info(f"Recovered from parent checkpoint {parent_id}")
                return parent

        # Strategy 2: Rebuild from event log
        events = await self._get_events_since_checkpoint(thread_id, parent_id)
        if events:
            rebuilt = await self._rebuild_from_events(parent, events)
            if self._validate_checkpoint(rebuilt):
                logger.info("Rebuilt checkpoint from event log")
                return rebuilt

        # Strategy 3: Return minimal valid state
        logger.warning("Full recovery failed, returning minimal state")
        return self._create_minimal_state(thread_id)

    def _create_minimal_state(self, thread_id: str) -> dict:
        \"\"\"Create minimal valid state as last resort\"\"\"
        return {
            "session_id": thread_id,
            "messages": [],
            "requirements": [],
            "confidence": 1.0,
            "current_turn": 0,
            "error_count": 0,
            "_recovered": True
        }
```

---

**Continuing with Production Readiness Checklist in final file...**
