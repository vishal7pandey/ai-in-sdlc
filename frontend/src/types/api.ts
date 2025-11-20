export interface SessionResponse {
  id: string
  project_name: string
  user_id: string
  status: string
  created_at: string
  updated_at: string | null
}

export type ApprovalStatus = 'pending' | 'approved' | 'revision_requested'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  metadata?: Record<string, unknown> | null
}

export interface Requirement {
  id: string
  title: string
  type: string
  actor: string
  action: string
  condition?: string | null
  acceptance_criteria: string[]
  priority?: string
  confidence: number
  inferred: boolean
  rationale?: string
  source_refs: string[]
}

export interface GraphState {
  session_id: string
  project_name: string
  user_id: string
  chat_history: Message[]
  current_turn: number
  requirements: Requirement[]
  inferred_requirements?: Requirement[]
  validation_issues?: Record<string, unknown>[]
  confidence?: number
  rd_draft: string | null
  rd_version?: number
  approval_status: ApprovalStatus
  review_feedback?: string | null
  last_agent?: string
  iterations?: number
  error_count?: number
}

export interface SessionDetailResponse extends SessionResponse {
  state: GraphState | null
}

export interface OrchestratorTurnResponse {
  status: 'ok' | 'interrupt'
  interrupt_type: string | null
  state: GraphState
}

export interface RDResponse {
  session_id: string
  version: number
  content: string
  format: 'markdown' | 'json' | 'pdf'
  status: 'draft' | 'under_review' | 'approved'
}

export interface ConnectionEstablishedEvent {
  type: 'connection.established'
  session_id: string
  server_time: string
  protocol_version: string
  features: {
    streaming: boolean
    heartbeat_interval: number
    max_message_size: number
  }
}

export interface MessageChunkEvent {
  type: 'message.chunk'
  session_id: string
  message_id: string
  turn_number: number
  delta: string
  accumulated?: string
  metadata: {
    model: string
    tokens_used: number
    confidence?: number
  }
  timestamp: string
}

export interface MessageCompleteEvent {
  type: 'message.complete'
  session_id: string
  message_id: string
  turn_number: number
  full_content: string
  metadata: {
    model: string
    total_tokens: number
    duration_ms: number
    confidence: number
    next_action?: string
  }
  timestamp: string
}

export interface RequirementSummary {
  id: string
  title: string
  type: string
  confidence: number
  inferred: boolean
}

export interface RequirementsExtractedEvent {
  type: 'requirements.extracted'
  session_id: string
  extraction_id: string
  requirements: RequirementSummary[]
  metadata: {
    extraction_duration_ms: number
    total_extracted: number
    total_session_requirements: number
  }
  timestamp: string
}

export interface AgentStatusEvent {
  type: 'agent.status'
  session_id: string
  agent: 'conversational' | 'extraction' | 'inference' | 'validation' | 'synthesis' | 'review'
  status: 'started' | 'running' | 'completed' | 'failed'
  metadata?: {
    duration_ms?: number
    error?: string
    progress?: number
  }
  timestamp: string
}

export interface PongEvent {
  type: 'pong'
  session_id: string
  timestamp: string
  server_time: string
}

export interface ErrorEvent {
  type: 'error'
  session_id: string
  error_code: string
  error_message: string
  severity: 'warning' | 'error' | 'fatal'
  metadata?: {
    agent?: string
    retry_after?: number
  }
  timestamp: string
}

export type ServerMessage =
  | ConnectionEstablishedEvent
  | MessageChunkEvent
  | MessageCompleteEvent
  | RequirementsExtractedEvent
  | AgentStatusEvent
  | PongEvent
  | ErrorEvent
