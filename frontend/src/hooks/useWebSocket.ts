import { useCallback, useEffect, useRef, useState } from 'react'

import type {
  AgentStatusEvent,
  ErrorEvent,
  MessageChunkEvent,
  MessageCompleteEvent,
  RequirementsExtractedEvent,
  ServerMessage,
} from '../types/api'

interface UseWebSocketProps {
  sessionId: string
  userId?: string
  onMessageChunk?: (chunk: MessageChunkEvent) => void
  onMessageComplete?: (message: MessageCompleteEvent) => void
  onRequirementsExtracted?: (data: RequirementsExtractedEvent) => void
  onAgentStatus?: (status: AgentStatusEvent) => void
  onError?: (error: ErrorEvent) => void
}

const HEARTBEAT_INTERVAL = 30000
const HEARTBEAT_TIMEOUT = 5000
const INITIAL_RETRY_DELAY = 1000
const MAX_RETRY_DELAY = 30000
const MAX_RECONNECT_ATTEMPTS = 10

export function useWebSocket({
  sessionId,
  userId = 'anonymous',
  onMessageChunk,
  onMessageComplete,
  onRequirementsExtracted,
  onAgentStatus,
  onError,
}: UseWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionQuality, setConnectionQuality] = useState<'good' | 'degraded' | 'poor'>('good')

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const heartbeatTimerRef = useRef<number | undefined>(undefined)
  const heartbeatTimeoutRef = useRef<number | undefined>(undefined)
  const messageQueueRef = useRef<unknown[]>([])

  const clearHeartbeatTimers = () => {
    if (heartbeatTimerRef.current !== undefined) {
      window.clearInterval(heartbeatTimerRef.current)
      heartbeatTimerRef.current = undefined
    }
    if (heartbeatTimeoutRef.current !== undefined) {
      window.clearTimeout(heartbeatTimeoutRef.current)
      heartbeatTimeoutRef.current = undefined
    }
  }

  const handlePong = () => {
    if (heartbeatTimeoutRef.current !== undefined) {
      window.clearTimeout(heartbeatTimeoutRef.current)
      heartbeatTimeoutRef.current = undefined
    }
    setConnectionQuality('good')
  }

  const sendRaw = useCallback((message: unknown) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message))
    } else {
      messageQueueRef.current.push(message)
    }
  }, [])

  const startHeartbeat = useCallback(() => {
    clearHeartbeatTimers()

    heartbeatTimerRef.current = window.setInterval(() => {
      sendRaw({
        type: 'ping',
        session_id: sessionId,
        timestamp: new Date().toISOString(),
      })

      heartbeatTimeoutRef.current = window.setTimeout(() => {
        setConnectionQuality('degraded')
        wsRef.current?.close(1006, 'Heartbeat timeout')
      }, HEARTBEAT_TIMEOUT)
    }, HEARTBEAT_INTERVAL)
  }, [sendRaw, sessionId])

  const scheduleReconnect = useCallback(() => {
    if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
      return
    }

    const delay = Math.min(INITIAL_RETRY_DELAY * 2 ** reconnectAttemptsRef.current, MAX_RETRY_DELAY)

    window.setTimeout(() => {
      reconnectAttemptsRef.current += 1
      connect()
    }, delay)
  }, [])

  const connect = useCallback(() => {
    const baseUrl = import.meta.env.VITE_WS_URL as string | undefined
    if (!baseUrl) {
      console.error('VITE_WS_URL is not configured')
      return
    }

    const wsUrl = `${baseUrl}?session_id=${sessionId}&user_id=${userId}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setIsConnected(true)
      setConnectionQuality('good')
      reconnectAttemptsRef.current = 0

      startHeartbeat()

      while (messageQueueRef.current.length > 0) {
        const message = messageQueueRef.current.shift()
        if (message) {
          ws.send(JSON.stringify(message))
        }
      }
    }

    ws.onmessage = (event: MessageEvent<string>) => {
      let data: ServerMessage
      try {
        data = JSON.parse(event.data) as ServerMessage
      } catch (err) {
        console.error('Failed to parse WebSocket message', err)
        return
      }

      switch (data.type) {
        case 'connection.established': {
          break
        }
        case 'message.chunk': {
          onMessageChunk?.(data)
          break
        }
        case 'message.complete': {
          onMessageComplete?.(data)
          break
        }
        case 'requirements.extracted': {
          onRequirementsExtracted?.(data)
          break
        }
        case 'agent.status': {
          onAgentStatus?.(data)
          break
        }
        case 'pong': {
          handlePong()
          break
        }
        case 'error': {
          onError?.(data)
          break
        }
        default: {
          break
        }
      }
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      setConnectionQuality('poor')
    }

    ws.onclose = (event) => {
      setIsConnected(false)
      clearHeartbeatTimers()

      if (event.code === 1000) {
        return
      }

      if (event.code >= 4000 && event.code < 4100) {
        return
      }

      scheduleReconnect()
    }

    wsRef.current = ws
  }, [onAgentStatus, onError, onMessageChunk, onMessageComplete, onRequirementsExtracted, scheduleReconnect, sessionId, startHeartbeat, userId])

  useEffect(() => {
    if (!sessionId) return

    connect()

    return () => {
      wsRef.current?.close(1000, 'Component unmounted')
      wsRef.current = null
      clearHeartbeatTimers()
    }
  }, [connect, sessionId])

  const sendChatMessage = useCallback(
    (content: string) => {
      if (!content.trim()) return

      const message = {
        type: 'chat.message' as const,
        session_id: sessionId,
        message_id: crypto.randomUUID(),
        content,
        timestamp: new Date().toISOString(),
      }

      sendRaw(message)
    },
    [sendRaw, sessionId],
  )

  return {
    isConnected,
    connectionQuality,
    sendChatMessage,
  }
}
