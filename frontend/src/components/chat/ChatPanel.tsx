import { useEffect, useState } from 'react'
import type { GraphState, Message } from '../../types/api'
import { getSessionDetail } from '../../lib/api/client'
import { useOnlineStatus } from '../../hooks/useOnlineStatus'
import { useWebSocket } from '../../hooks/useWebSocket'

interface ChatPanelProps {
  sessionId: string
  state: GraphState | null
  onStateUpdate: (state: GraphState) => void
}

export function ChatPanel({ sessionId, state, onStateUpdate }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [queuedMessages, setQueuedMessages] = useState<string[]>([])
  const [messages, setMessages] = useState<Message[]>(state?.chat_history ?? [])
  const [streamingMessage, setStreamingMessage] = useState<Message | null>(null)
  const isOnline = useOnlineStatus()

  useEffect(() => {
    if (state?.chat_history) {
      setMessages(state.chat_history)
    }
  }, [state])

  const { sendChatMessage } = useWebSocket({
    sessionId,
    onMessageChunk: (chunk) => {
      setError(null)
      setIsSending(false)
      setStreamingMessage((prev) => {
        if (prev && prev.id === chunk.message_id) {
          return { ...prev, content: prev.content + chunk.delta }
        }
        return {
          id: chunk.message_id,
          role: 'assistant',
          content: chunk.delta,
          timestamp: chunk.timestamp,
          metadata: null,
        }
      })
    },
    onMessageComplete: (message) => {
      setStreamingMessage(null)
      setIsSending(false)
      setMessages((prev) => [
        ...prev,
        {
          id: message.message_id,
          role: 'assistant',
          content: message.full_content,
          timestamp: message.timestamp,
          metadata: null,
        },
      ])
    },
    onRequirementsExtracted: async (_event) => {
      try {
        const detail = await getSessionDetail(sessionId)
        if (detail.state) {
          onStateUpdate(detail.state)
        }
      } catch (err) {
        console.error('Failed to refresh session after requirements extraction', err)
      }
    },
    onError: (err) => {
      setIsSending(false)
      setError(err.error_message)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || isSending) return

    // If offline, queue the message locally to be sent later.
    if (!isOnline) {
      setQueuedMessages((prev) => [...prev, text])
      setInput('')
      return
    }

    setIsSending(true)
    setError(null)

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
      metadata: null,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    sendChatMessage(text)
  }

  // When connection is restored, flush any queued messages sequentially.
  useEffect(() => {
    if (!isOnline || queuedMessages.length === 0 || isSending) return

    let cancelled = false

    const flushQueue = () => {
      setIsSending(true)
      setError(null)

      try {
        for (const text of queuedMessages) {
          if (cancelled) break
          const trimmed = text.trim()
          if (!trimmed) continue

          const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: trimmed,
            timestamp: new Date().toISOString(),
            metadata: null,
          }

          setMessages((prev) => [...prev, userMessage])
          sendChatMessage(trimmed)
        }

        if (!cancelled) {
          setQueuedMessages([])
        }
      } catch (err) {
        console.error(err)
        if (!cancelled) {
          setError('Failed to send some queued messages')
        }
      } finally {
        if (!cancelled) {
          setIsSending(false)
        }
      }
    }

    flushQueue()

    return () => {
      cancelled = true
    }
  }, [isOnline, queuedMessages, isSending, sendChatMessage])

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {!isOnline && (
        <div
          className="px-4 py-2 text-xs text-amber-300 bg-amber-950 border-b border-amber-800"
          role="status"
          aria-live="polite"
        >
          You appear to be offline. Messages cannot be sent until the connection is restored.
        </div>
      )}
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {messages.length === 0 && !streamingMessage && queuedMessages.length === 0 && (
          <p className="text-sm text-slate-500">
            Start the conversation by sending a message about your project requirements.
          </p>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              data-testid="chat-message"
              data-role={msg.role}
              className={`max-w-[70%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-sky-600 text-slate-50'
                  : 'bg-slate-800 text-slate-100'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {streamingMessage && (
          <div className="flex justify-start">
            <div
              data-testid="streaming-message"
              className="max-w-[70%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap bg-slate-800 text-slate-100"
            >
              {streamingMessage.content}
            </div>
          </div>
        )}

        {queuedMessages.map((text, index) => (
          <div key={`queued-${index}`} className="flex justify-end opacity-70">
            <div className="max-w-[70%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap bg-slate-700 text-slate-100 border border-dashed border-slate-400">
              {text}
              <div className="mt-1 text-[10px] text-slate-300">Queued (offline)</div>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="px-4 py-2 text-xs text-rose-400 bg-rose-950 border-t border-rose-900">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="p-3 border-t border-slate-800 flex gap-2">
        <input
          className="flex-1 rounded bg-slate-950 border border-slate-700 px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-500"
          placeholder="Describe a feature or requirement..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isSending || (!isOnline && queuedMessages.length >= 10)}
        />
        <button
          type="submit"
          disabled={isSending || (!isOnline && queuedMessages.length >= 10)}
          className="text-sm px-3 py-2 rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-slate-50"
          aria-label="Send message"
        >
          Send
        </button>
      </form>
    </div>
  )
}
