import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import type { GraphState } from '../types/api'
import { getSessionDetail } from '../lib/api/client'
import { ChatPanel } from '../components/chat/ChatPanel'
import { RequirementsSidebar } from '../components/requirements/RequirementsSidebar'

export function SessionPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const [state, setState] = useState<GraphState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mobileView, setMobileView] = useState<'chat' | 'requirements'>('chat')

  useEffect(() => {
    if (!sessionId) return

    let cancelled = false

    void (async () => {
      try {
        const detail = await getSessionDetail(sessionId)
        if (cancelled) return
        setState(detail.state)
        setError(null)
      } catch (err: unknown) {
        if (cancelled) return
        console.error(err)
        setError('Failed to load session')
      }
    })()

    return () => {
      cancelled = true
    }
  }, [sessionId])

  const isLoading = !!sessionId && state === null && !error

  if (!sessionId) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-slate-500">
        No session selected.
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col md:flex-row md:divide-x divide-slate-800">
      <section className="flex-1 min-h-0 flex flex-col bg-slate-900">
        <div className="p-4 border-b border-slate-800 text-sm font-medium text-slate-300 flex items-center justify-between">
          <span>Chat</span>
          {isLoading && <span className="text-xs text-slate-500">Loading</span>}
          {error && <span className="text-xs text-rose-400">{error}</span>}
        </div>
        {/* Mobile toggle between Chat and Requirements */}
        <div className="flex md:hidden border-b border-slate-800 text-xs">
          <button
            type="button"
            onClick={() => setMobileView('chat')}
            className={`flex-1 px-3 py-2 text-center border-r border-slate-800 ${
              mobileView === 'chat' ? 'bg-slate-900 text-slate-100' : 'bg-slate-950 text-slate-500'
            }`}
          >
            Chat
          </button>
          <button
            type="button"
            onClick={() => setMobileView('requirements')}
            className={`flex-1 px-3 py-2 text-center ${
              mobileView === 'requirements'
                ? 'bg-slate-900 text-slate-100'
                : 'bg-slate-950 text-slate-500'
            }`}
          >
            Requirements
          </button>
        </div>
        <div className="flex-1 min-h-0">
          {/* On mobile, show only the selected view; on md+ both are visible via the layout below */}
          {mobileView === 'chat' && (
            <div className="md:hidden h-full">
              <ChatPanel sessionId={sessionId} state={state} onStateUpdate={setState} />
            </div>
          )}
          {mobileView === 'requirements' && (
            <div className="md:hidden h-full">
              <RequirementsSidebar sessionId={sessionId} state={state} />
            </div>
          )}

          {/* Desktop/tablet: Chat panel lives here; requirements live in the right-hand sidebar */}
          <div className="hidden md:block h-full">
            <ChatPanel sessionId={sessionId} state={state} onStateUpdate={setState} />
          </div>
        </div>
      </section>
      <div className="hidden md:block">
        <RequirementsSidebar sessionId={sessionId} state={state} />
      </div>
    </div>
  )
}
