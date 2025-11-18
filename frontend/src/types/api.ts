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
