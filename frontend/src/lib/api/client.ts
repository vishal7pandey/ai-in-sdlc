import type { SessionDetailResponse, OrchestratorTurnResponse, RDResponse } from '../../types/api'

export const API_BASE = '/api/v1'

function getUserId(): string {
  if (typeof window === 'undefined') {
    return 'frontend-demo'
  }
  return window.localStorage.getItem('userId') ?? 'frontend-demo'
}

export async function apiFetch<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'X-User-Id': getUserId(),
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`)
  }

  return response.json() as Promise<TResponse>
}

export async function getSessionDetail(sessionId: string): Promise<SessionDetailResponse> {
  return apiFetch<SessionDetailResponse>(`/sessions/${sessionId}`)
}

export async function sendSessionMessage(
  sessionId: string,
  message: string,
  projectName?: string,
): Promise<OrchestratorTurnResponse> {
  return apiFetch<OrchestratorTurnResponse>(`/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: JSON.stringify({ message, project_name: projectName ?? undefined }),
  })
}

export async function generateRD(sessionId: string): Promise<RDResponse> {
  return apiFetch<RDResponse>(`/rd/${sessionId}/generate`, {
    method: 'POST',
  })
}

export async function getRD(sessionId: string): Promise<RDResponse> {
  return apiFetch<RDResponse>(`/rd/${sessionId}`)
}

export async function exportRD(
  sessionId: string,
): Promise<{ filename: string; content: string }> {
  return apiFetch<{ filename: string; content: string }>(`/rd/${sessionId}/export`)
}
