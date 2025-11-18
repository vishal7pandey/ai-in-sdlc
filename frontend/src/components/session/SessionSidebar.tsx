import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessionStore } from '../../store/sessionStore'

export function SessionSidebar() {
  const navigate = useNavigate()

  const sessions = useSessionStore((s) => s.sessions)
  const currentSession = useSessionStore((s) => s.currentSession)
  const isLoading = useSessionStore((s) => s.isLoading)
  const loadSessions = useSessionStore((s) => s.loadSessions)
  const createSession = useSessionStore((s) => s.createSession)
  const setCurrentSession = useSessionStore((s) => s.setCurrentSession)

  useEffect(() => {
    void loadSessions()
  }, [loadSessions])

  const handleCreateClick = async () => {
    const name = window.prompt('Project name')
    if (!name) return

    const session = await createSession(name)
    if (session) {
      navigate(`/sessions/${session.id}`)
    }
  }

  const handleSelect = (sessionId: string) => {
    const session = sessions.find((s) => s.id === sessionId)
    if (!session) return
    setCurrentSession(session)
    navigate(`/sessions/${session.id}`)
  }

  return (
    <aside className="hidden md:flex w-72 border-r border-slate-800 bg-slate-950/95 flex-col p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-200">Sessions</h2>
        <button
          type="button"
          className="text-xs px-2 py-1 rounded bg-sky-600 hover:bg-sky-500 text-slate-50"
          onClick={handleCreateClick}
        >
          New
        </button>
      </div>

      {isLoading && (
        <p className="text-xs text-slate-500">Loading sessions</p>
      )}

      {!isLoading && sessions.length === 0 && (
        <p className="text-xs text-slate-500">
          No sessions yet. Click <span className="font-semibold">New</span> to create one.
        </p>
      )}

      {!isLoading && sessions.length > 0 && (
        <div className="flex-1 overflow-auto space-y-1 text-sm">
          {sessions.map((session) => {
            const isActive = currentSession?.id === session.id
            return (
              <button
                key={session.id}
                type="button"
                onClick={() => handleSelect(session.id)}
                className={
                  'w-full text-left px-3 py-2 rounded border text-xs ' +
                  (isActive
                    ? 'bg-slate-800 border-sky-500 text-slate-50'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-900')
                }
              >
                <div className="font-medium truncate">{session.projectName}</div>
                <div className="text-[10px] text-slate-500 flex justify-between">
                  <span>{session.status}</span>
                  <span>{new Date(session.createdAt).toLocaleDateString()}</span>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </aside>
  )
}
