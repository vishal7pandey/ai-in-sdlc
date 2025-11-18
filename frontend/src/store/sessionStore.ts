import { create } from 'zustand'
import type { SessionResponse } from '../types/api'

export interface Session {
  id: string
  projectName: string
  status: string
  createdAt: string
  updatedAt: string | null
}

interface SessionState {
  currentSession: Session | null
  sessions: Session[]
  isLoading: boolean
  error?: string
  setCurrentSession: (session: Session) => void
  loadSessions: () => Promise<void>
  createSession: (projectName: string) => Promise<Session | null>
}

const API_BASE = '/api/v1'

function getUserId(): string {
  if (typeof window === 'undefined') {
    return 'frontend-demo'
  }
  return window.localStorage.getItem('userId') ?? 'frontend-demo'
}

async function apiFetch(path: string, init?: RequestInit) {
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

  return response.json()
}

export const useSessionStore = create<SessionState>((set) => ({
  currentSession: null,
  sessions: [],
  isLoading: false,
  error: undefined,

  setCurrentSession: (session) => set({ currentSession: session }),

  loadSessions: async () => {
    set({ isLoading: true, error: undefined })
    try {
      const data = (await apiFetch('/sessions')) as SessionResponse[]
      const sessions: Session[] = data.map((row) => ({
        id: row.id,
        projectName: row.project_name,
        status: row.status,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      }))
      set({ sessions, isLoading: false })
    } catch (error) {
      console.error('Failed to load sessions', error)
      set({ isLoading: false, error: (error as Error).message })
    }
  },

  createSession: async (projectName: string) => {
    if (!projectName.trim()) {
      return null
    }

    set({ isLoading: true, error: undefined })
    try {
      const payload = { project_name: projectName.trim() }
      const row = await apiFetch('/sessions', {
        method: 'POST',
        body: JSON.stringify(payload),
      })

      const session: Session = {
        id: row.id,
        projectName: row.project_name,
        status: row.status,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      }

      set((state) => ({
        sessions: [...state.sessions, session],
        currentSession: session,
        isLoading: false,
      }))

      return session
    } catch (error) {
      console.error('Failed to create session', error)
      set({ isLoading: false, error: (error as Error).message })
      return null
    }
  },
}))
