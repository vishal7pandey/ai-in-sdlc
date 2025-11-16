# Design Packet 4: Part 2 - Observability, CI/CD, Security & Production Readiness

## 2. Observability & Monitoring Architecture

### 2.1 Backend Observability

#### 2.1.1 Structured Logging Schema[126][129][132]

```python
# src/utils/observability.py
import logging
import json
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")

class ObservabilityLogger:
    \"\"\"Structured logging for observability\"\"\"

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

    def log(
        self,
        level: str,
        message: str,
        agent: str = None,
        event_type: str = None,
        **extra_fields
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "level": level,
            "message": message,
            "request_id": request_id_var.get(""),
            "session_id": session_id_var.get(""),
            "user_id": user_id_var.get(""),
        }

        if agent:
            log_entry["agent"] = agent

        if event_type:
            log_entry["event_type"] = event_type

        # Add custom fields
        log_entry.update(extra_fields)

        # Log as JSON
        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(log_entry)
        )

# Agent-specific logging
class AgentLogger(ObservabilityLogger):
    def log_agent_start(self, agent_name: str, state: Dict):
        self.log(
            "info",
            f"{agent_name} started",
            agent=agent_name,
            event_type="agent_start",
            turn=state.get("current_turn"),
            requirements_count=len(state.get("requirements", []))
        )

    def log_agent_complete(self, agent_name: str, duration_ms: float, result: Dict):
        self.log(
            "info",
            f"{agent_name} completed",
            agent=agent_name,
            event_type="agent_complete",
            duration_ms=duration_ms,
            confidence=result.get("confidence"),
            requirements_added=len(result.get("requirements", []))
        )

    def log_agent_error(self, agent_name: str, error: Exception):
        self.log(
            "error",
            f"{agent_name} failed",
            agent=agent_name,
            event_type="agent_error",
            error_type=type(error).__name__,
            error_message=str(error)
        )

    def log_llm_call(
        self,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ):
        self.log(
            "info",
            "LLM call completed",
            agent=agent_name,
            event_type="llm_call",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms
        )
```

#### 2.1.2 Distributed Tracing (LangGraph Nodes)

```python
# src/utils/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from functools import wraps
import time

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter (or use OTLP for generic backends)
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

def trace_agent_node(agent_name: str):
    \"\"\"Decorator to trace LangGraph node execution\"\"\"
    def decorator(func):
        @wraps(func)
        async def wrapper(state, *args, **kwargs):
            with tracer.start_as_current_span(
                f"agent.{agent_name}",
                attributes={
                    "agent.name": agent_name,
                    "session.id": state.get("session_id"),
                    "turn": state.get("current_turn"),
                    "requirements.count": len(state.get("requirements", []))
                }
            ) as span:
                start_time = time.time()

                try:
                    result = await func(state, *args, **kwargs)

                    # Add result attributes
                    span.set_attribute("agent.success", True)
                    span.set_attribute("confidence", result.get("confidence", 0))
                    span.set_attribute(
                        "requirements.added",
                        len(result.get("requirements", [])) - len(state.get("requirements", []))
                    )

                    return result

                except Exception as e:
                    span.set_attribute("agent.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise

                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

        return wrapper
    return decorator

# Usage in agent
@trace_agent_node("extraction")
async def extraction_node(state: GraphState) -> GraphState:
    # Agent logic
    pass

# Trace LLM calls separately
def trace_llm_call(model: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                "llm.call",
                attributes={"llm.model": model}
            ) as span:
                start_time = time.time()

                response = await func(*args, **kwargs)

                # Extract token usage
                if hasattr(response, 'usage'):
                    span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                    span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
                    span.set_attribute("llm.total_tokens", response.usage.total_tokens)

                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.latency_ms", duration_ms)

                return response

        return wrapper
    return decorator
```

#### 2.1.3 Prometheus Metrics[126][129][132][135]

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from functools import wraps
import time

# Agent metrics
agent_invocations_total = Counter(
    'agent_invocations_total',
    'Total number of agent invocations',
    ['agent_name', 'status']  # Labels: extraction, inference, etc. / success, error
)

agent_duration_seconds = Histogram(
    'agent_duration_seconds',
    'Agent execution duration in seconds',
    ['agent_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

agent_confidence_score = Histogram(
    'agent_confidence_score',
    'Agent confidence scores',
    ['agent_name'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# LLM metrics
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total LLM tokens consumed',
    ['model', 'token_type']  # token_type: prompt, completion
)

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']  # status: success, error, timeout
)

llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM API call latency',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)

llm_cost_usd = Counter(
    'llm_cost_usd',
    'Estimated LLM cost in USD',
    ['model']
)

# Requirements metrics
requirements_extracted_total = Counter(
    'requirements_extracted_total',
    'Total requirements extracted',
    ['type', 'inferred']  # type: functional, non-functional, etc.
)

requirements_validation_issues_total = Counter(
    'requirements_validation_issues_total',
    'Total validation issues detected',
    ['severity', 'issue_type']  # severity: error, warning, info
)

# Database metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Current database connection pool size'
)

db_optimistic_lock_conflicts_total = Counter(
    'db_optimistic_lock_conflicts_total',
    'Total optimistic locking conflicts',
    ['table']
)

# WebSocket metrics
websocket_connections_active = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)

websocket_messages_total = Counter(
    'websocket_messages_total',
    'Total WebSocket messages',
    ['direction', 'message_type']  # direction: inbound, outbound
)

websocket_errors_total = Counter(
    'websocket_errors_total',
    'Total WebSocket errors',
    ['error_type']
)

# Session metrics
sessions_active = Gauge(
    'sessions_active',
    'Number of active sessions'
)

sessions_created_total = Counter(
    'sessions_created_total',
    'Total sessions created'
)

rd_documents_generated_total = Counter(
    'rd_documents_generated_total',
    'Total RD documents generated',
    ['format']  # format: pdf, markdown, json
)

# System health
system_info = Info(
    'system_info',
    'System information'
)

# Decorator for agent metrics
def track_agent_metrics(agent_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)

                # Track confidence
                if "confidence" in result:
                    agent_confidence_score.labels(agent_name=agent_name).observe(
                        result["confidence"]
                    )

                # Track requirements extracted
                if "requirements" in result and "requirements" in args[0]:
                    new_reqs = len(result["requirements"]) - len(args[0]["requirements"])
                    for req in result["requirements"][-new_reqs:]:
                        requirements_extracted_total.labels(
                            type=req.type,
                            inferred=str(req.inferred)
                        ).inc()

                return result

            except Exception as e:
                status = "error"
                raise

            finally:
                duration = time.time() - start_time
                agent_invocations_total.labels(
                    agent_name=agent_name,
                    status=status
                ).inc()
                agent_duration_seconds.labels(agent_name=agent_name).observe(duration)

        return wrapper
    return decorator

# LLM call tracking
def track_llm_call(model: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                response = await func(*args, **kwargs)

                # Track token usage
                if hasattr(response, 'usage'):
                    llm_tokens_total.labels(
                        model=model,
                        token_type="prompt"
                    ).inc(response.usage.prompt_tokens)

                    llm_tokens_total.labels(
                        model=model,
                        token_type="completion"
                    ).inc(response.usage.completion_tokens)

                    # Estimate cost (example rates)
                    cost = estimate_llm_cost(
                        model,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                    llm_cost_usd.labels(model=model).inc(cost)

                return response

            except TimeoutError:
                status = "timeout"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                llm_requests_total.labels(model=model, status=status).inc()
                llm_latency_seconds.labels(model=model).observe(duration)

        return wrapper
    return decorator

def estimate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    \"\"\"Estimate LLM API cost in USD\"\"\"
    # Example pricing (update with actual rates)
    pricing = {
        "gpt-4-turbo-preview": {
            "prompt": 0.01 / 1000,  # $0.01 per 1K tokens
            "completion": 0.03 / 1000
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0015 / 1000,
            "completion": 0.002 / 1000
        }
    }

    rates = pricing.get(model, {"prompt": 0, "completion": 0})
    return (prompt_tokens * rates["prompt"]) + (completion_tokens * rates["completion"])
```

#### 2.1.4 Alerting Rules (Prometheus)[126][129][132][135]

```yaml
# prometheus/alerts.yml
groups:
  - name: agent_alerts
    interval: 30s
    rules:
      # Agent performance
      - alert: AgentHighLatency
        expr: histogram_quantile(0.95, rate(agent_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Agent {{ $labels.agent_name }} has high latency"
          description: "95th percentile latency is {{ $value }}s (threshold: 30s)"
          runbook: "https://docs.internal/runbooks/agent-latency"

      - alert: AgentHighErrorRate
        expr: |
          sum(rate(agent_invocations_total{status="error"}[5m])) by (agent_name)
          /
          sum(rate(agent_invocations_total[5m])) by (agent_name)
          > 0.05
        for: 10m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "Agent {{ $labels.agent_name }} error rate above 5%"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: AgentLowConfidence
        expr: avg_over_time(agent_confidence_score[15m]) < 0.6
        for: 15m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "Agent {{ $labels.agent_name }} confidence dropping"
          description: "Average confidence is {{ $value }}"

      # LLM alerts
      - alert: LLMHighLatency
        expr: histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "LLM API latency high for {{ $labels.model }}"
          description: "95th percentile latency is {{ $value }}s"

      - alert: LLMRateLimitApproaching
        expr: rate(llm_requests_total{status="error"}[1m]) > 5
        for: 2m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "LLM rate limit may be approaching"
          description: "Error rate: {{ $value }} req/s"

      - alert: LLMCostSpike
        expr: |
          rate(llm_cost_usd[1h])
          >
          avg_over_time(rate(llm_cost_usd[24h])[7d:1h]) * 2
        for: 15m
        labels:
          severity: warning
          team: finance
        annotations:
          summary: "LLM costs spiking for {{ $labels.model }}"
          description: "Cost rate 2x higher than 7-day average"

      # Database alerts
      - alert: DatabaseSlowQueries
        expr: histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Slow database queries on {{ $labels.table }}"
          description: "95th percentile query time is {{ $value }}s"

      - alert: OptimisticLockingConflicts
        expr: rate(db_optimistic_lock_conflicts_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "High optimistic locking conflicts on {{ $labels.table }}"
          description: "Conflict rate: {{ $value }} /s"

      - alert: DatabaseConnectionPoolExhaustion
        expr: db_connection_pool_size > 90
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Pool size: {{ $value }}"

      # WebSocket alerts
      - alert: WebSocketHighDisconnectRate
        expr: rate(websocket_errors_total{error_type="disconnect"}[5m]) > 1
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "High WebSocket disconnect rate"
          description: "Disconnect rate: {{ $value }} /s"

      - alert: WebSocketMessageBacklog
        expr: |
          rate(websocket_messages_total{direction="inbound"}[1m])
          >
          rate(websocket_messages_total{direction="outbound"}[1m]) * 1.5
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "WebSocket message backlog building up"
          description: "Inbound rate exceeds outbound by 50%"

      # Extraction quality alerts
      - alert: RequirementExtractionAccuracyDrop
        expr: |
          sum(rate(requirements_validation_issues_total{severity="error"}[15m]))
          /
          sum(rate(requirements_extracted_total[15m]))
          > 0.2
        for: 15m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "Requirement extraction quality degrading"
          description: "{{ $value | humanizePercentage }} of requirements have errors"

      # System health
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "High memory usage"
          description: "Memory usage at {{ $value | humanizePercentage }}"
```

#### 2.1.5 Dashboards (Grafana)

```json
// grafana/dashboards/agent_performance.json
{
  "dashboard": {
    "title": "Agent Performance Dashboard",
    "panels": [
      {
        "title": "Agent Invocations (Rate)",
        "targets": [{
          "expr": "sum(rate(agent_invocations_total[5m])) by (agent_name)"
        }],
        "type": "graph"
      },
      {
        "title": "Agent Latency (P95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(agent_duration_seconds_bucket[5m])) by (agent_name)"
        }],
        "type": "graph"
      },
      {
        "title": "Agent Error Rate",
        "targets": [{
          "expr": "sum(rate(agent_invocations_total{status=\"error\"}[5m])) by (agent_name) / sum(rate(agent_invocations_total[5m])) by (agent_name)"
        }],
        "type": "graph"
      },
      {
        "title": "Agent Confidence Scores",
        "targets": [{
          "expr": "avg_over_time(agent_confidence_score[5m]) by (agent_name)"
        }],
        "type": "graph"
      },
      {
        "title": "LLM Token Usage",
        "targets": [{
          "expr": "sum(rate(llm_tokens_total[5m])) by (model, token_type)"
        }],
        "type": "graph"
      },
      {
        "title": "LLM Cost (Hourly Rate)",
        "targets": [{
          "expr": "sum(rate(llm_cost_usd[1h])) by (model) * 3600"
        }],
        "type": "singlestat"
      },
      {
        "title": "Requirements Extracted (by Type)",
        "targets": [{
          "expr": "sum(rate(requirements_extracted_total[5m])) by (type)"
        }],
        "type": "piechart"
      },
      {
        "title": "WebSocket Connections",
        "targets": [{
          "expr": "websocket_connections_active"
        }],
        "type": "singlestat"
      }
    ]
  }
}
```

### 2.2 Frontend Observability

#### 2.2.1 User Performance Telemetry

```typescript
// src/lib/telemetry.ts
import { onCLS, onFID, onLCP, onFCP, onTTFB } from 'web-vitals';

interface PerformanceMetric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  timestamp: number;
}

class TelemetryService {
  private endpoint = '/api/v1/telemetry';

  init() {
    // Core Web Vitals
    onCLS(this.sendMetric.bind(this));  // Cumulative Layout Shift
    onFID(this.sendMetric.bind(this));  // First Input Delay
    onLCP(this.sendMetric.bind(this));  // Largest Contentful Paint
    onFCP(this.sendMetric.bind(this));  // First Contentful Paint
    onTTFB(this.sendMetric.bind(this)); // Time to First Byte

    // Custom metrics
    this.trackCustomMetrics();
  }

  private sendMetric(metric: any) {
    const data: PerformanceMetric = {
      name: metric.name,
      value: metric.value,
      rating: metric.rating,
      timestamp: Date.now()
    };

    // Send to backend (use beacon for reliability)
    if (navigator.sendBeacon) {
      navigator.sendBeacon(this.endpoint, JSON.stringify(data));
    } else {
      fetch(this.endpoint, {
        method: 'POST',
        body: JSON.stringify(data),
        keepalive: true
      });
    }
  }

  private trackCustomMetrics() {
    // Time to Interactive
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'measure') {
            this.sendMetric({
              name: entry.name,
              value: entry.duration,
              rating: entry.duration < 3000 ? 'good' : 'poor'
            });
          }
        }
      });

      observer.observe({ entryTypes: ['measure'] });
    }

    // Track route changes
    this.trackRouteChanges();
  }

  private trackRouteChanges() {
    let lastRoute = window.location.pathname;

    setInterval(() => {
      const currentRoute = window.location.pathname;
      if (currentRoute !== lastRoute) {
        performance.mark('route-change-start');

        // Wait for route to stabilize
        requestIdleCallback(() => {
          performance.mark('route-change-end');
          performance.measure(
            'route-change-duration',
            'route-change-start',
            'route-change-end'
          );
        });

        lastRoute = currentRoute;
      }
    }, 100);
  }

  // Track user interactions
  trackInteraction(action: string, metadata?: Record<string, any>) {
    fetch(this.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'interaction',
        action,
        metadata,
        timestamp: Date.now(),
        url: window.location.pathname
      })
    });
  }
}

export const telemetry = new TelemetryService();
```

#### 2.2.2 Error Boundary Instrumentation

```typescript
// src/components/ErrorBoundary.tsx
import React, { Component, ErrorInfo } from 'react';

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log to observability service
    this.logError(error, errorInfo);
  }

  private logError(error: Error, errorInfo: ErrorInfo) {
    fetch('/api/v1/errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error: {
          message: error.message,
          stack: error.stack,
          name: error.name
        },
        componentStack: errorInfo.componentStack,
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString()
      })
    }).catch(console.error);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <details>
            <summary>Error details</summary>
            <pre>{this.state.error?.message}</pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}
```

---

## 3. CI/CD Architecture

### 3.1 Local Development CI (Git Hooks)

```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

echo "🔍 Running pre-commit checks..."

# Backend checks
echo "Checking Python code..."
cd backend
poetry run black --check src tests || exit 1
poetry run ruff check src tests || exit 1
poetry run mypy src || exit 1

# Frontend checks
echo "Checking TypeScript code..."
cd ../frontend
npm run lint || exit 1
npm run typecheck || exit 1

echo "✅ Pre-commit checks passed"
```

```bash
# .husky/pre-push
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

echo "🧪 Running tests before push..."

# Backend tests
cd backend
poetry run pytest tests/unit -v --cov=src --cov-report=term-missing || exit 1

# Frontend tests
cd ../frontend
npm run test:unit || exit 1

echo "✅ All tests passed"
```

### 3.2 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  backend-test:
    name: Backend Tests
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: reqeng_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

      chromadb:
        image: chromadb/chroma:latest
        ports:
          - 8001:8000

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache Poetry dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pypoetry
          key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: |
          cd backend
          poetry install

      - name: Run linting
        run: |
          cd backend
          poetry run black --check src tests
          poetry run ruff check src tests
          poetry run mypy src

      - name: Run unit tests
        run: |
          cd backend
          poetry run pytest tests/unit -v --cov=src --cov-report=xml --cov-report=term-missing

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/reqeng_test
          REDIS_URL: redis://localhost:6379
          CHROMA_URL: http://localhost:8001
        run: |
          cd backend
          poetry run pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml
          flags: backend

  frontend-test:
    name: Frontend Tests
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: |
          cd frontend
          npm ci

      - name: Run linting
        run: |
          cd frontend
          npm run lint
          npm run format:check

      - name: Run type checking
        run: |
          cd frontend
          npm run typecheck

      - name: Run unit tests
        run: |
          cd frontend
          npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./frontend/coverage/coverage-final.json
          flags: frontend

  e2e-test:
    name: E2E Tests
    needs: [backend-test, frontend-test]
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: reqeng_test
      redis:
        image: redis:7
      chromadb:
        image: chromadb/chroma:latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python & Node
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - uses: actions/setup-node@v3
        with:
          node-version: ${{ env.NODE_VERSION }}

      - name: Install dependencies
        run: |
          cd backend && poetry install
          cd ../frontend && npm ci

      - name: Start backend
        run: |
          cd backend
          poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 &
          sleep 10

      - name: Start frontend
        run: |
          cd frontend
          npm run build
          npm run preview &
          sleep 5

      - name: Install Playwright
        run: |
          cd frontend
          npx playwright install --with-deps

      - name: Run E2E tests
        run: |
          cd frontend
          npm run test:e2e

      - name: Upload Playwright report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: playwright-report
          path: frontend/playwright-report/

  llm-prompt-regression:
    name: LLM Prompt Regression Tests
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          cd backend
          poetry install

      - name: Run golden tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd backend
          poetry run pytest tests/golden -v --golden-regen=false

      - name: Check for prompt drift
        run: |
          cd backend
          # Compare current prompts with baseline
          poetry run python scripts/check_prompt_drift.py

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Bandit (Python security linter)
        run: |
          cd backend
          poetry add --group dev bandit
          poetry run bandit -r src -f json -o bandit-report.json

      - name: Run npm audit
        run: |
          cd frontend
          npm audit --audit-level=high
```

### 3.3 Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build-and-push:
    name: Build & Push Docker Images
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push backend
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/backend:latest
            ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
          cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/backend:buildcache
          cache-to: type=registry,ref=ghcr.io/${{ github.repository }}/backend:buildcache,mode=max

      - name: Build and push frontend
        uses: docker/build-push-action@v4
        with:
          context: ./frontend
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/frontend:latest
            ghcr.io/${{ github.repository }}/frontend:${{ github.sha }}
          cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/frontend:buildcache
          cache-to: type=registry,ref=ghcr.io/${{ github.repository }}/frontend:buildcache,mode=max

  deploy-local:
    name: Deploy to Local Environment
    needs: build-and-push
    runs-on: self-hosted

    steps:
      - uses: actions/checkout@v3

      - name: Pull latest images
        run: |
          docker pull ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
          docker pull ghcr.io/${{ github.repository }}/frontend:${{ github.sha }}

      - name: Update docker-compose
        run: |
          export BACKEND_IMAGE=ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
          export FRONTEND_IMAGE=ghcr.io/${{ github.repository }}/frontend:${{ github.sha }}
          docker-compose -f docker-compose.prod.yml up -d

      - name: Wait for health check
        run: |
          sleep 10
          curl -f http://localhost:8000/health || exit 1

      - name: Run smoke tests
        run: |
          ./scripts/smoke-tests.sh
```

---

**Continuing with Security, Reliability, and Production Readiness in next file...**
