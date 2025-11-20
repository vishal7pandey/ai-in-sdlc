import { useState } from 'react'

import type { GraphState, Requirement } from '../../types/api'
import { exportRD, generateRD } from '../../lib/api/client'

interface RequirementsSidebarProps {
  sessionId: string
  state: GraphState | null
}

export function RequirementsSidebar({ sessionId, state }: RequirementsSidebarProps) {
  const requirements: Requirement[] = state?.requirements ?? []
  const [rdContent, setRdContent] = useState<string | null>(state?.rd_draft ?? null)

  const handleGenerate = async () => {
    if (!requirements.length) return

    try {
      const rd = await generateRD(sessionId)
      setRdContent(rd.content)
    } catch (error) {
      console.error('Failed to generate RD draft', error)
    }
  }

  const handleExport = async () => {
    if (!rdContent) return

    const download = (filename: string, content: string) => {
      const blob = new Blob([content], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    }

    try {
      const { filename, content } = await exportRD(sessionId)
      download(filename, content)
    } catch (error) {
      console.error('Failed to export RD via API, falling back to local draft', error)
      try {
        download(`requirements-${sessionId}.md`, rdContent)
      } catch (fallbackError) {
        // Soft-fail; keep this in console only for now.
        console.error('Failed to export RD markdown', fallbackError)
      }
    }
  }

  return (
    <aside className="w-96 flex flex-col min-h-0 bg-slate-950/90 border-l border-slate-800">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">Requirements</h2>
        <span className="text-[11px] text-slate-500">
          {requirements.length} item{requirements.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-3 text-sm">
        {requirements.length === 0 && (
          <p className="text-slate-500">
            Extracted requirements will appear here as you converse with the AI.
          </p>
        )}

        {requirements.map((req) => (
          <div
            key={req.id}
            className="border border-slate-800 rounded-lg p-3 bg-slate-900/80 hover:border-sky-500 transition-colors"
          >
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs font-semibold text-slate-300 truncate">
                {req.id} · {req.title}
              </div>
              <div className="text-[10px] text-slate-500 uppercase">{req.type}</div>
            </div>
            <div className="text-xs text-slate-400 mb-1">
              <span className="font-semibold">Actor:</span> {req.actor} ·{' '}
              <span className="font-semibold">Action:</span> {req.action}
            </div>
            {(() => {
              const confidence = req.confidence ?? 0
              const percentage = Math.round(confidence * 100)
              const barColor =
                confidence >= 0.8
                  ? 'bg-emerald-500'
                  : confidence >= 0.6
                  ? 'bg-amber-400'
                  : 'bg-rose-500'

              return (
                <div className="mt-1">
                  <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span className="font-semibold">Confidence</span>
                    <span>{percentage}%</span>
                  </div>
                  <div className="mt-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full ${barColor}`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              )
            })()}
          </div>
        ))}
      </div>

      <div className="border-t border-slate-800 p-3 h-40 flex flex-col bg-slate-950/95">
        <div className="flex items-center justify-between mb-2">
          <div className="font-semibold text-slate-300 text-xs">RD Preview</div>
          <div className="flex items-center gap-2">
            {requirements.length > 0 && (
              <button
                type="button"
                onClick={handleGenerate}
                className="text-[11px] px-2 py-1 rounded border border-slate-700 text-slate-200 hover:bg-slate-800"
              >
                Generate RD
              </button>
            )}
            {rdContent && (
              <button
                type="button"
                onClick={handleExport}
                className="text-[11px] px-2 py-1 rounded border border-slate-700 text-slate-200 hover:bg-slate-800"
              >
                Export MD
              </button>
            )}
          </div>
        </div>
        <div className="flex-1 overflow-auto text-[11px] text-slate-400">
          {rdContent ? (
            <pre className="whitespace-pre-wrap">{rdContent}</pre>
          ) : (
            <p className="text-slate-500">
              Once the orchestrator generates a Requirements Document draft, it will appear here.
            </p>
          )}
        </div>
      </div>
    </aside>
  )
}
