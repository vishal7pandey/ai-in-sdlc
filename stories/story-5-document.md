# User Story 5: Frontend MVP - React Chat Interface with Real-Time Requirements Display

## Story Overview

**Story ID:** STORY-005
**Story Title:** Frontend MVP - Interactive Chat UI with Requirements Visualization
**Priority:** P0 - Critical (First User-Facing Interface)
**Estimated Effort:** 28-36 hours
**Sprint:** Sprint 5
**Dependencies:**
- STORY-001 (Project Foundation) - Complete ✅
- STORY-002 (Conversational Agent) - Complete ✅
- STORY-003 (Extraction Agent) - Complete ✅
- STORY-004 (LangGraph Orchestrator) - Complete ✅

---

## Socratic Design Questions & Answers

### Q1: What is the core problem the UI solves?
**A:** The backend works but is **invisible** to users. Engineers and stakeholders need a **visual, interactive way** to:
- Have natural conversations with the AI
- See requirements extracted in real-time
- Understand AI confidence levels
- Review and approve generated requirements
- Export the Requirements Document

Without UI: Users must use curl/Postman (developer-only)
With UI: Anyone can gather requirements through conversation

### Q2: What makes this an "MVP" versus a "full UI"?
**A:** MVP focuses on **core workflow** only:
- ✅ Session creation and management
- ✅ Chat interface with AI responses
- ✅ Real-time requirement cards display
- ✅ Confidence scoring visualization
- ✅ Requirements Document preview
- ✅ Basic export (Markdown)

**NOT in MVP** (future stories):
- ❌ Inline editing of requirements
- ❌ Comment threads and collaboration
- ❌ Version diff viewer
- ❌ Multi-stakeholder approval workflow
- ❌ Advanced filtering and search
- ❌ PDF export with templates

### Q3: What technology stack ensures rapid development?
**A:** **Modern React ecosystem** for speed and reliability:
- **React 18** + **TypeScript** - Type safety, latest features
- **Vite 5** - Instant HMR, 10x faster than Webpack
- **TailwindCSS 3** - Utility-first styling, no CSS overhead
- **shadcn/ui** - Pre-built accessible components (copy-paste)
- **Zustand** - Minimal state management (< 1KB)
- **React Query** - API state with caching
- **WebSocket** - Real-time updates

**Why NOT Next.js/Remix?** MVP doesn't need SSR/SSG. SPA is faster to develop.

### Q4: How does the UI communicate with the backend?
**A:** **Dual-channel architecture**:

```
REST API (Read-heavy operations):
- GET /api/v1/sessions - List sessions
- POST /api/v1/sessions - Create session
- GET /api/v1/sessions/{id}/requirements - Get requirements
- GET /api/v1/rd/{session_id} - Get RD

WebSocket (Write-heavy + streaming):
- User sends message → WS: {type: 'chat.message'}
- Backend streams response → WS: {type: 'message.chunk'}
- Backend extracts requirements → WS: {type: 'requirements.extracted'}
- Agent status updates → WS: {type: 'agent.update'}
```

**Why both?** REST for simple queries, WebSocket for real-time bidirectional communication.

### Q5: What are the core UX flows in the MVP?
**A:** **Three primary workflows**:

1. **Session Start Flow** (30 seconds)
   ```
   User clicks "New Session"
   → Enters project name
   → Session created
   → Chat panel auto-focuses
   ```

2. **Conversation Flow** (2-3 minutes per requirement)
   ```
   User types requirement
   → AI responds with clarifying questions
   → User provides details
   → Requirement card appears on right panel
   → Confidence score shows (0-100%)
   → User continues conversation or reviews
   ```

3. **Review & Export Flow** (1 minute)
   ```
   User reviews requirement cards
   → Clicks "Generate Document"
   → RD preview appears
   → User clicks "Export Markdown"
   → File downloads
   ```

### Q6: How do we visualize AI confidence and reasoning?
**A:** **Progressive disclosure** approach:
- **High confidence (80%+)**: Green progress bar, minimal UI
- **Medium confidence (60-80%)**: Yellow bar, "Review suggested" hint
- **Low confidence (<60%)**: Orange bar, warning icon, "Clarification needed"

**Traceability**: Each requirement card shows:
- Source chat turns (clickable links)
- Extraction timestamp
- Agent that created it (conversational vs inference)

### Q7: What does "real-time" mean in this context?
**A:** **Sub-second feedback** for user actions:
- **Typing message**: Instant echo (optimistic update)
- **AI response**: Streams character-by-character (< 100ms latency)
- **Requirement extraction**: Card fades in within 500ms
- **Agent status**: Pills update in real-time (idle → running → complete)

**Technical implementation**:
- Optimistic UI updates
- WebSocket with 30s heartbeat
- Reconnection with exponential backoff
- Offline queue for messages

### Q8: How do we handle errors gracefully?
**A:** **Four-tier error handling**:

1. **Network errors**: "Offline" banner, queue messages, auto-reconnect
2. **API errors**: Toast notifications with retry button
3. **Validation errors**: Inline field errors, prevent submission
4. **LLM failures**: "AI temporarily unavailable, manual entry available"

**User never loses data**: Auto-save to localStorage every 10s.

### Q9: What accessibility standards must we meet?
**A:** **WCAG 2.1 AA compliance** (industry standard):
- Keyboard navigation (Tab, Enter, Escape)
- Screen reader support (ARIA labels)
- Color contrast ratio ≥ 4.5:1
- Focus indicators visible
- No information conveyed by color alone

**Testing**: `axe-core` automated checks in CI.

### Q10: What defines "done" for this MVP?
**A:** **Functional completeness**:
- ✅ User can complete end-to-end workflow (create → chat → extract → export)
- ✅ All core components render without errors
- ✅ WebSocket connection stable with reconnection
- ✅ Responsive design (desktop + tablet, mobile optional)
- ✅ No console errors or warnings
- ✅ Load time < 2 seconds on 3G
- ✅ Lighthouse score ≥ 85 (performance, accessibility, best practices)

**Demo-ready**: Can show to stakeholders without disclaimers.

---

## Story Description

As a **requirements engineer**, I want a **visual, web-based interface where I can chat with the AI, see requirements extracted in real-time, and export the Requirements Document** so that **I can efficiently gather requirements from stakeholders without using command-line tools or API clients**.

This story implements the **first user-facing interface** for the platform. The MVP includes:

- **Session Management**: Create new sessions, view session history, resume previous sessions
- **Chat Interface**: Type messages, see streaming AI responses, markdown support
- **Requirements Display**: Real-time requirement cards with confidence scores, traceability
- **RD Preview**: Generated Requirements Document in readable format
- **Export**: Download RD as Markdown file
- **Real-Time Updates**: WebSocket connection for instant feedback, agent status indicators
- **Error Handling**: Graceful offline mode, reconnection logic, user-friendly error messages

**Key Outcome**: After this story, **any user can interact with the platform through a browser**, gathering requirements through natural conversation with immediate visual feedback.

---

## Business Value

- **Democratizes access**: Non-technical users can now use the platform
- **Accelerates adoption**: Visual interface reduces learning curve by 80%
- **Enables demos**: Can showcase to investors, customers, and stakeholders
- **Validates UX**: Real user testing becomes possible
- **Reduces support burden**: Self-explanatory UI reduces "how do I use this?" questions
- **Competitive advantage**: Professional UI differentiates from CLI-only tools

---

## Acceptance Criteria

### ✅ AC1: Project Setup and Tooling

**Given** a new frontend directory
**When** I set up the project
**Then** all tooling is configured for rapid development:

**Directory Structure:**
```
frontend/
├── public/
│   ├── favicon.ico
│   └── robots.txt
├── src/
│   ├── assets/
│   │   └── icons/
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   └── TypingIndicator.tsx
│   │   ├── requirements/
│   │   │   ├── RequirementCard.tsx
│   │   │   ├── RequirementsList.tsx
│   │   │   └── ConfidenceScore.tsx
│   │   ├── rd-viewer/
│   │   │   ├── RDViewer.tsx
│   │   │   └── ExportMenu.tsx
│   │   ├── session/
│   │   │   ├── SessionSidebar.tsx
│   │   │   ├── SessionCard.tsx
│   │   │   └── CreateSessionModal.tsx
│   │   ├── layout/
│   │   │   ├── AppLayout.tsx
│   │   │   ├── Header.tsx
│   │   │   └── Sidebar.tsx
│   │   └── ui/          # shadcn/ui components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── dialog.tsx
│   │       ├── input.tsx
│   │       └── ...
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useSession.ts
│   │   └── useRequirements.ts
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   ├── sessions.ts
│   │   │   └── requirements.ts
│   │   ├── websocket/
│   │   │   ├── client.ts
│   │   │   └── events.ts
│   │   └── utils.ts
│   ├── store/
│   │   ├── sessionStore.ts
│   │   ├── chatStore.ts
│   │   └── requirementsStore.ts
│   ├── types/
│   │   ├── api.ts
│   │   ├── session.ts
│   │   └── requirement.ts
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   └── SessionPage.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── .env.example
├── .eslintrc.cjs
├── .prettierrc
├── components.json       # shadcn config
├── tailwind.config.ts
├── tsconfig.json
├── vite.config.ts
└── package.json
```

**Package.json:**
```json
{
  "name": "reqeng-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx",
    "format": "prettier --write src/**/*.{ts,tsx}"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.14.2",
    "zustand": "^4.4.7",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-toast": "^1.1.5",
    "@radix-ui/react-progress": "^1.0.3",
    "lucide-react": "^0.294.0",
    "date-fns": "^2.30.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.1.0",
    "zod": "^3.22.4"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.8",
    "typescript": "^5.2.2",
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "eslint": "^8.55.0",
    "prettier": "^3.1.1",
    "@typescript-eslint/eslint-plugin": "^6.14.0"
  }
}
```

**Verification:**
```bash
cd frontend
npm install
npm run dev

# Expected: Vite dev server starts on http://localhost:5173
# Expected: No TypeScript errors
# Expected: Hot module replacement working
```

---

### ✅ AC2: Layout and Navigation Structure

**Given** the app is loaded
**When** user visits the application
**Then** a clean, professional layout is displayed:

**Layout Specification:**
```
┌────────────────────────────────────────────────────────────────┐
│  Header: Requirements Engineering Platform        [User Menu]  │
├───────────┬─────────────────────────────┬──────────────────────┤
│           │                             │                      │
│  Session  │     Chat Panel              │  Requirements Panel  │
│  Sidebar  │                             │                      │
│           │  [User Message]             │  ┌─────────────────┐ │
│ ┌───────┐ │  [AI Response]              │  │ REQ-001     89% │ │
│ │Active │ │  [User Message]             │  │ User login      │ │
│ │Sess 1 │ │  [Typing indicator...]      │  │ with email...   │ │
│ └───────┘ │                             │  └─────────────────┘ │
│           │                             │                      │
│ ┌───────┐ │                             │  ┌─────────────────┐ │
│ │Sess 2 │ │  [Chat Input Field]         │  │ REQ-002     75% │ │
│ │Draft  │ │  [Send Button]              │  │ Performance...  │ │
│ └───────┘ │                             │  └─────────────────┘ │
│           │                             │                      │
│ [New +]   │                             │  [Generate RD]       │
│           │                             │                      │
└───────────┴─────────────────────────────┴──────────────────────┘
```

**Component Implementation:**
```tsx
// src/components/layout/AppLayout.tsx
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { SessionSidebar } from '../session/SessionSidebar';

export function AppLayout() {
  return (
    <div className="h-screen flex flex-col">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <SessionSidebar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

**Verification:**
```bash
npm run dev
# Open http://localhost:5173
# Expected: Layout renders with header, sidebar, main content area
# Expected: Responsive (sidebar collapses on mobile)
```

---

### ✅ AC3: Session Management

**Given** the app is loaded
**When** user interacts with sessions
**Then** they can create, view, and switch between sessions:

**Session Store (Zustand):**
```tsx
// src/store/sessionStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Session {
  id: string;
  projectName: string;
  status: 'active' | 'draft' | 'archived';
  createdAt: string;
  requirementsCount: number;
  rdVersion: number;
}

interface SessionState {
  currentSession: Session | null;
  sessions: Session[];
  isLoading: boolean;

  setCurrentSession: (session: Session) => void;
  loadSessions: () => Promise<void>;
  createSession: (projectName: string) => Promise<Session>;
  deleteSession: (sessionId: string) => Promise<void>;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      currentSession: null,
      sessions: [],
      isLoading: false,

      setCurrentSession: (session) => set({ currentSession: session }),

      loadSessions: async () => {
        set({ isLoading: true });
        try {
          const response = await fetch('/api/v1/sessions');
          const data = await response.json();
          set({ sessions: data.items, isLoading: false });
        } catch (error) {
          console.error('Failed to load sessions:', error);
          set({ isLoading: false });
        }
      },

      createSession: async (projectName) => {
        set({ isLoading: true });
        const response = await fetch('/api/v1/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ projectName }),
        });
        const session = await response.json();
        set((state) => ({
          sessions: [...state.sessions, session],
          currentSession: session,
          isLoading: false,
        }));
        return session;
      },

      deleteSession: async (sessionId) => {
        await fetch(`/api/v1/sessions/${sessionId}`, { method: 'DELETE' });
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== sessionId),
          currentSession: state.currentSession?.id === sessionId ? null : state.currentSession,
        }));
      },
    }),
    { name: 'session-storage' }
  )
);
```

**Create Session Modal:**
```tsx
// src/components/session/CreateSessionModal.tsx
import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { useSessionStore } from '../../store/sessionStore';

export function CreateSessionModal({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [projectName, setProjectName] = useState('');
  const createSession = useSessionStore((s) => s.createSession);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!projectName.trim()) return;

    await createSession(projectName);
    onOpenChange(false);
    setProjectName('');
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create New Session</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium">Project Name</label>
            <Input
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="E.g., E-commerce Platform v2"
              autoFocus
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!projectName.trim()}>
              Create
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
```

**Verification:**
```bash
# In browser:
# 1. Click "New Session" button
# 2. Enter project name "Test Project"
# 3. Click Create
# Expected: Modal closes, new session appears in sidebar, chat panel loads
# Expected: Session persisted to localStorage
# Expected: API call made to POST /api/v1/sessions
```

---

### ✅ AC4: Chat Interface with Message Display

**Given** an active session
**When** user sends messages
**Then** messages display correctly with streaming responses:

**Chat Store:**
```tsx
// src/store/chatStore.ts
import { create } from 'zustand';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    confidence?: number;
    requirementsExtracted?: number;
  };
}

interface ChatState {
  messages: Message[];
  streamingMessage: string;
  isStreaming: boolean;

  addMessage: (message: Message) => void;
  setStreamingMessage: (content: string) => void;
  finalizeStreamingMessage: (id: string) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  streamingMessage: '',
  isStreaming: false,

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setStreamingMessage: (content) =>
    set({ streamingMessage: content, isStreaming: true }),

  finalizeStreamingMessage: (id) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          id,
          role: 'assistant',
          content: state.streamingMessage,
          timestamp: new Date().toISOString(),
        },
      ],
      streamingMessage: '',
      isStreaming: false,
    })),

  clearMessages: () => set({ messages: [], streamingMessage: '' }),
}));
```

**Message Bubble Component:**
```tsx
// src/components/chat/MessageBubble.tsx
import { format } from 'date-fns';
import { cn } from '../../lib/utils';
import { Badge } from '../ui/badge';

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    confidence?: number;
    requirementsExtracted?: number;
  };
}

export function MessageBubble({ role, content, timestamp, metadata }: MessageBubbleProps) {
  const isUser = role === 'user';

  return (
    <div className={cn('flex gap-3 mb-4', isUser && 'flex-row-reverse')}>
      {/* Avatar */}
      <div
        className={cn(
          'w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
        )}
      >
        {isUser ? 'You' : 'AI'}
      </div>

      {/* Message content */}
      <div className="flex-1 max-w-[70%]">
        <div
          className={cn(
            'rounded-lg px-4 py-2',
            isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
          )}
        >
          <div className="prose prose-sm max-w-none">{content}</div>
        </div>

        {/* Timestamp and metadata */}
        <div className="flex items-center justify-between mt-2 text-xs opacity-70">
          <span>{format(new Date(timestamp), 'h:mm a')}</span>

          {metadata && (
            <div className="flex gap-2">
              {metadata.requirementsExtracted && (
                <Badge variant="secondary" className="text-xs">
                  {metadata.requirementsExtracted} requirements extracted
                </Badge>
              )}
              {metadata.confidence && (
                <Badge variant="outline" className="text-xs">
                  {Math.round(metadata.confidence * 100)}% confidence
                </Badge>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Chat Panel:**
```tsx
// src/components/chat/ChatPanel.tsx
import { useEffect, useRef } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { TypingIndicator } from './TypingIndicator';

export function ChatPanel({ sessionId }: { sessionId: string }) {
  const messages = useChatStore((s) => s.messages);
  const streamingMessage = useChatStore((s) => s.streamingMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const addMessage = useChatStore((s) => s.addMessage);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { sendMessage, isConnected } = useWebSocket({ sessionId });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  const handleSendMessage = (text: string) => {
    // Optimistic update
    addMessage({
      id: `temp-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    });

    // Send via WebSocket
    sendMessage('chat.message', {
      sessionId,
      message: text,
      timestamp: new Date().toISOString(),
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Connection status */}
      {!isConnected && (
        <div className="bg-yellow-100 text-yellow-900 px-4 py-2 text-sm">
          Reconnecting to server...
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} {...msg} />
        ))}

        {isStreaming && (
          <MessageBubble
            role="assistant"
            content={streamingMessage}
            timestamp={new Date().toISOString()}
          />
        )}

        {isStreaming && <TypingIndicator />}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={handleSendMessage} disabled={!isConnected} />
    </div>
  );
}
```

**Verification:**
```bash
# In browser:
# 1. Type message "We need a login feature"
# 2. Press Enter
# Expected: Message appears instantly on right
# Expected: Typing indicator shows
# Expected: AI response streams in character-by-character
# Expected: Message auto-scrolls to bottom
```

---

### ✅ AC5: WebSocket Integration

**Given** the app is running
**When** WebSocket events occur
**Then** UI updates in real-time:

**WebSocket Hook:**
```tsx
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { useRequirementsStore } from '../store/requirementsStore';

interface UseWebSocketProps {
  sessionId: string;
}

export function useWebSocket({ sessionId }: UseWebSocketProps) {
  const socketRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectAttempts = useRef(0);

  const setStreamingMessage = useChatStore((s) => s.setStreamingMessage);
  const finalizeStreamingMessage = useChatStore((s) => s.finalizeStreamingMessage);
  const addRequirements = useRequirementsStore((s) => s.addRequirements);

  const connect = useCallback(() => {
    const ws = new WebSocket(
      `${import.meta.env.VITE_WS_URL}?sessionId=${sessionId}`
    );

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'message.chunk':
          if (data.payload.isFinal) {
            finalizeStreamingMessage(data.payload.messageId);
          } else {
            setStreamingMessage((prev) => prev + data.payload.content);
          }
          break;

        case 'requirements.extracted':
          addRequirements(data.payload.requirements);
          break;

        case 'agent.update':
          // Update agent status indicator
          console.log('Agent update:', data.payload);
          break;

        case 'error':
          console.error('WebSocket error:', data.payload);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket disconnected');
      setIsConnected(false);

      // Reconnect with exponential backoff
      if (event.code !== 1000) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectAttempts.current++;
        setTimeout(connect, delay);
      }
    };

    socketRef.current = ws;
  }, [sessionId]);

  useEffect(() => {
    connect();
    return () => {
      socketRef.current?.close(1000);
    };
  }, [connect]);

  const sendMessage = useCallback((type: string, payload: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type, payload }));
    }
  }, []);

  return { isConnected, sendMessage };
}
```

**Verification:**
```bash
# In browser DevTools (Network tab):
# 1. Check WS connection established
# 2. Send message "Test"
# Expected: WS frame sent with type "chat.message"
# Expected: Multiple WS frames received with type "message.chunk"
# Expected: Final frame with isFinal: true
# Expected: UI updates smoothly during streaming
```

---

### ✅ AC6: Requirements Panel with Cards

**Given** requirements are extracted
**When** they appear in the panel
**Then** they display with confidence scores and details:

**Requirements Store:**
```tsx
// src/store/requirementsStore.ts
import { create } from 'zustand';

interface Requirement {
  id: string;
  title: string;
  type: string;
  confidence: number;
  inferred: boolean;
  actor: string;
  action: string;
  acceptanceCriteria: string[];
  sourceRefs: string[];
}

interface RequirementsState {
  requirements: Requirement[];
  addRequirements: (reqs: Requirement[]) => void;
  updateRequirement: (id: string, updates: Partial<Requirement>) => void;
  removeRequirement: (id: string) => void;
}

export const useRequirementsStore = create<RequirementsState>((set) => ({
  requirements: [],

  addRequirements: (reqs) =>
    set((state) => ({ requirements: [...state.requirements, ...reqs] })),

  updateRequirement: (id, updates) =>
    set((state) => ({
      requirements: state.requirements.map((r) =>
        r.id === id ? { ...r, ...updates } : r
      ),
    })),

  removeRequirement: (id) =>
    set((state) => ({
      requirements: state.requirements.filter((r) => r.id !== id),
    })),
}));
```

**Requirement Card:**
```tsx
// src/components/requirements/RequirementCard.tsx
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { ConfidenceScore } from './ConfidenceScore';

interface RequirementCardProps {
  requirement: Requirement;
  onClick?: () => void;
}

export function RequirementCard({ requirement, onClick }: RequirementCardProps) {
  return (
    <Card
      className="cursor-pointer hover:shadow-md transition-shadow"
      onClick={onClick}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            {requirement.id}
            <span className="font-normal text-muted-foreground">
              {requirement.title}
            </span>
            {requirement.inferred && (
              <Badge variant="secondary" className="text-xs">
                INFERRED
              </Badge>
            )}
          </CardTitle>
        </div>
        <ConfidenceScore value={requirement.confidence} />
      </CardHeader>

      <CardContent className="pt-0 space-y-2">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="font-medium">Type:</span>
            <Badge variant="outline">{requirement.type}</Badge>
          </div>
          <div>
            <span className="font-medium">Actor:</span> {requirement.actor}
          </div>
        </div>

        <div className="text-sm">
          <span className="font-medium">Action:</span> {requirement.action}
        </div>

        <div className="text-sm">
          <span className="font-medium">Criteria:</span>
          <ul className="list-disc list-inside text-muted-foreground mt-1">
            {requirement.acceptanceCriteria.slice(0, 2).map((c, i) => (
              <li key={i}>{c}</li>
            ))}
            {requirement.acceptanceCriteria.length > 2 && (
              <li className="text-xs">
                +{requirement.acceptanceCriteria.length - 2} more
              </li>
            )}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Confidence Score Component:**
```tsx
// src/components/requirements/ConfidenceScore.tsx
import { Progress } from '../ui/progress';
import { cn } from '../../lib/utils';

export function ConfidenceScore({ value }: { value: number }) {
  const percentage = Math.round(value * 100);
  const color =
    value >= 0.8
      ? 'text-green-600'
      : value >= 0.6
      ? 'text-yellow-600'
      : 'text-orange-600';

  return (
    <div className="flex items-center gap-2">
      <Progress
        value={percentage}
        className="h-2 w-24"
        indicatorClassName={cn(
          value >= 0.8 && 'bg-green-600',
          value >= 0.6 && value < 0.8 && 'bg-yellow-600',
          value < 0.6 && 'bg-orange-600'
        )}
      />
      <span className={cn('text-xs font-medium', color)}>{percentage}%</span>
    </div>
  );
}
```

**Requirements Panel:**
```tsx
// src/components/requirements/RequirementsPanel.tsx
import { useRequirementsStore } from '../../store/requirementsStore';
import { RequirementCard } from './RequirementCard';
import { Button } from '../ui/button';

export function RequirementsPanel() {
  const requirements = useRequirementsStore((s) => s.requirements);

  return (
    <div className="w-96 border-l flex flex-col">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold">Requirements</h2>
        <p className="text-sm text-muted-foreground">
          {requirements.length} extracted
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {requirements.length === 0 ? (
          <div className="text-center text-muted-foreground mt-8">
            <p>No requirements yet.</p>
            <p className="text-sm mt-2">
              Start a conversation to extract requirements.
            </p>
          </div>
        ) : (
          requirements.map((req) => (
            <RequirementCard key={req.id} requirement={req} />
          ))
        )}
      </div>

      {requirements.length > 0 && (
        <div className="p-4 border-t">
          <Button className="w-full">Generate RD</Button>
        </div>
      )}
    </div>
  );
}
```

**Verification:**
```bash
# In browser:
# 1. Send message that triggers extraction
# 2. Wait for AI to extract requirements
# Expected: Requirement card fades in on right panel
# Expected: Confidence score shows as colored progress bar
# Expected: All requirement details visible
# Expected: Smooth animation on appearance
```

---

### ✅ AC7: RD Preview and Export

**Given** requirements are extracted
**When** user clicks "Generate RD"
**Then** RD preview displays with export option:

**RD Viewer Component:**
```tsx
// src/components/rd-viewer/RDViewer.tsx
import { useState } from 'react';
import { Button } from '../ui/button';
import { Download } from 'lucide-react';

interface RDViewerProps {
  sessionId: string;
  requirements: Requirement[];
}

export function RDViewer({ sessionId, requirements }: RDViewerProps) {
  const [rdContent, setRdContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const generateRD = async () => {
    setIsLoading(true);
    const response = await fetch(`/api/v1/rd/${sessionId}/generate`, {
      method: 'POST',
    });
    const data = await response.json();
    setRdContent(data.content);
    setIsLoading(false);
  };

  const exportMarkdown = () => {
    const blob = new Blob([rdContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `requirements-${sessionId}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!rdContent) {
    return (
      <div className="flex items-center justify-center h-full">
        <Button onClick={generateRD} disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate Requirements Document'}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="text-xl font-semibold">Requirements Document</h2>
        <Button onClick={exportMarkdown} variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          Export Markdown
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-4xl mx-auto prose prose-sm">
          <pre className="whitespace-pre-wrap">{rdContent}</pre>
        </div>
      </div>
    </div>
  );
}
```

**Verification:**
```bash
# In browser:
# 1. Extract some requirements via conversation
# 2. Click "Generate RD" button
# Expected: Loading indicator appears
# Expected: RD content appears in preview after 2-3 seconds
# Expected: Markdown formatted nicely
# 3. Click "Export Markdown" button
# Expected: File downloads immediately
# Expected: Filename is "requirements-{session-id}.md"
```

---

### ✅ AC8: Error Handling and Offline Mode

**Given** the app is running
**When** network errors occur
**Then** graceful degradation happens:

**Offline Banner:**
```tsx
// src/components/ui/OfflineBanner.tsx
import { WifiOff } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from './alert';

export function OfflineBanner({ isOnline }: { isOnline: boolean }) {
  if (isOnline) return null;

  return (
    <Alert variant="warning" className="mb-4">
      <WifiOff className="h-4 w-4" />
      <AlertTitle>Offline Mode</AlertTitle>
      <AlertDescription>
        Your changes are saved locally and will sync when you're back online.
      </AlertDescription>
    </Alert>
  );
}
```

**Offline Hook:**
```tsx
// src/hooks/useOnlineStatus.ts
import { useEffect, useState } from 'react';
import { toast } from '../components/ui/use-toast';

export function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    function handleOnline() {
      setIsOnline(true);
      toast({ title: 'Back online', description: 'Reconnected to server' });
    }

    function handleOffline() {
      setIsOnline(false);
      toast({
        title: 'You are offline',
        description: 'Changes will sync when reconnected.',
        variant: 'warning',
      });
    }

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}
```

**Verification:**
```bash
# In browser DevTools:
# 1. Open Network tab
# 2. Select "Offline" throttling
# Expected: Banner appears "You are offline"
# Expected: Send button disabled
# Expected: Messages queued locally
# 3. Select "Online" throttling
# Expected: Banner disappears
# Expected: Queued messages sent
# Expected: Toast notification "Back online"
```

---

### ✅ AC9: Responsive Design

**Given** different screen sizes
**When** user views the app
**Then** layout adapts appropriately:

**Responsive Breakpoints:**
- **Desktop (≥1280px)**: Full 3-column layout
- **Tablet (768px-1279px)**: 2-column, sidebar collapsible
- **Mobile (<768px)**: Single column, bottom navigation

**Responsive Layout:**
```tsx
// src/components/layout/AppLayout.tsx (updated)
import { useState } from 'react';
import { Menu } from 'lucide-react';
import { Button } from '../ui/button';

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="h-screen flex flex-col">
      <Header>
        <Button
          variant="ghost"
          size="sm"
          className="md:hidden"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          <Menu className="h-5 w-5" />
        </Button>
      </Header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - collapsible on mobile */}
        <div
          className={cn(
            'transition-transform',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full',
            'md:translate-x-0',
            'absolute md:relative z-20 md:z-0'
          )}
        >
          <SessionSidebar />
        </div>

        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

**Verification:**
```bash
# In browser DevTools (Device toolbar):
# 1. View at 1920px width
# Expected: 3 columns visible (sidebar, chat, requirements)
# 2. View at 1024px width
# Expected: Sidebar still visible, requirements panel stacked
# 3. View at 375px width (iPhone)
# Expected: Single column, hamburger menu for sidebar
# Expected: Requirements accessible via tab/swipe
```

---

### ✅ AC10: Performance Optimization

**Given** the app is deployed
**When** performance is measured
**Then** it meets Lighthouse targets:

**Performance Targets:**
- Load time: < 2 seconds on 3G
- Time to interactive: < 3 seconds
- First contentful paint: < 1 second
- Lighthouse score: ≥ 85 (performance, accessibility, best practices)

**Optimizations Implemented:**
```tsx
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          'query-vendor': ['@tanstack/react-query'],
          'state-vendor': ['zustand'],
        },
      },
    },
    chunkSizeWarningLimit: 1000, // 1MB warning threshold
  },
});
```

**Lazy Loading:**
```tsx
// src/App.tsx
import { lazy, Suspense } from 'react';

const SessionPage = lazy(() => import('./pages/SessionPage'));
const RDViewer = lazy(() => import('./components/rd-viewer/RDViewer'));

export function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/session/:id" element={<SessionPage />} />
        <Route path="/rd/:id" element={<RDViewer />} />
      </Routes>
    </Suspense>
  );
}
```

**Verification:**
```bash
npm run build
npm run preview

# Run Lighthouse audit
npx lighthouse http://localhost:4173 --view

# Expected scores:
# Performance: ≥85
# Accessibility: ≥85
# Best Practices: ≥85
# SEO: ≥85
```

---

### ✅ AC11: Accessibility Compliance

**Given** the app is running
**When** accessibility is tested
**Then** WCAG 2.1 AA standards are met:

**Accessibility Features:**
- Keyboard navigation (Tab, Enter, Escape, Arrow keys)
- Screen reader support (ARIA labels, roles, live regions)
- Color contrast ratio ≥ 4.5:1
- Focus indicators visible on all interactive elements
- No reliance on color alone for information

**Implementation:**
```tsx
// Example: Accessible button
<Button
  aria-label="Send message"
  aria-describedby="chat-input-help"
  onKeyDown={(e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSend();
    }
  }}
>
  Send
</Button>

// Example: Live region for screen readers
<div
  role="status"
  aria-live="polite"
  aria-atomic="true"
  className="sr-only"
>
  {isConnected ? 'Connected to server' : 'Connecting...'}
</div>
```

**Verification:**
```bash
# Install axe DevTools browser extension
# Run accessibility scan
# Expected: 0 critical issues
# Expected: 0 serious issues
# Expected: All interactive elements keyboard-accessible

# Test with screen reader (VoiceOver on Mac, NVDA on Windows)
# Expected: All text readable
# Expected: Navigation logical
# Expected: Form labels present
```

---

### ✅ AC12: End-to-End User Flow Test

**Given** a fresh browser session
**When** user completes full workflow
**Then** all steps work seamlessly:

**E2E Test Scenario:**
```
1. User opens http://localhost:5173
   ✓ Homepage loads in < 2 seconds
   ✓ No console errors

2. User clicks "New Session"
   ✓ Modal opens
   ✓ Input is auto-focused

3. User types "E-commerce Platform"
   ✓ Text appears in input

4. User clicks "Create"
   ✓ Modal closes
   ✓ New session appears in sidebar
   ✓ Chat panel loads
   ✓ Input is auto-focused

5. User types "Users should log in with email and password"
   ✓ Message appears instantly on right (optimistic)
   ✓ WebSocket sends message
   ✓ Typing indicator appears

6. AI responds
   ✓ Response streams character-by-character
   ✓ Message auto-scrolls to bottom
   ✓ Typing indicator disappears

7. Requirements extracted
   ✓ Requirement card appears on right panel
   ✓ Confidence score shows (e.g., 89%)
   ✓ Smooth fade-in animation

8. User types "Also needs password reset"
   ✓ Second requirement extracted
   ✓ Two cards now visible

9. User clicks "Generate RD"
   ✓ Loading indicator shows
   ✓ RD content appears in preview
   ✓ Markdown formatted

10. User clicks "Export Markdown"
    ✓ File downloads immediately
    ✓ File contains both requirements
```

**Automated E2E Test (Playwright):**
```typescript
// tests/e2e/full-workflow.spec.ts
import { test, expect } from '@playwright/test';

test('Complete requirement gathering flow', async ({ page }) => {
  // Step 1: Load homepage
  await page.goto('http://localhost:5173');
  await expect(page).toHaveTitle(/Requirements Engineering/);

  // Step 2: Create session
  await page.click('text=New Session');
  await page.fill('input[placeholder*="Project Name"]', 'Test Project');
  await page.click('text=Create');

  // Step 3: Send message
  await page.fill('textarea[placeholder*="Type"]', 'Users need login');
  await page.press('textarea', 'Enter');

  // Step 4: Wait for AI response
  await expect(page.locator('text=AI')).toBeVisible({ timeout: 10000 });

  // Step 5: Wait for requirement card
  await expect(page.locator('text=REQ-001')).toBeVisible({ timeout: 15000 });

  // Step 6: Verify confidence score
  const confidenceScore = page.locator('.confidence-score');
  await expect(confidenceScore).toBeVisible();

  // Step 7: Generate RD
  await page.click('text=Generate RD');
  await expect(page.locator('text=Requirements Document')).toBeVisible({ timeout: 5000 });

  // Step 8: Export
  const downloadPromise = page.waitForEvent('download');
  await page.click('text=Export Markdown');
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/requirements-.*\.md/);
});
```

**Verification:**
```bash
npm run test:e2e

# Expected: All steps pass
# Expected: Total time < 30 seconds
# Expected: Screenshots captured on failure
```

---

## Technical Implementation Summary

### Core Files Created (30+ components)

| Category | Files | Lines |
|----------|-------|-------|
| **Layout** | AppLayout, Header, Sidebar | 400+ |
| **Session** | SessionSidebar, SessionCard, CreateSessionModal | 350+ |
| **Chat** | ChatPanel, MessageBubble, ChatInput, TypingIndicator | 500+ |
| **Requirements** | RequirementCard, RequirementsList, ConfidenceScore | 400+ |
| **RD Viewer** | RDViewer, ExportMenu | 250+ |
| **Hooks** | useWebSocket, useSession, useRequirements, useOnlineStatus | 400+ |
| **Store** | sessionStore, chatStore, requirementsStore | 350+ |
| **API** | client, sessions, requirements, websocket | 300+ |
| **UI Components** | button, card, dialog, input, progress, badge (shadcn) | 800+ |

**Total**: ~3,750 lines of TypeScript/React code

---

## Testing Strategy

### Unit Tests (20+ tests)
- Component rendering
- Store state management
- Utility functions
- Hook behavior

### Integration Tests (10+ tests)
- WebSocket event handling
- API calls with React Query
- Store updates from events
- Component interactions

### E2E Tests (5+ tests)
- Full user workflow
- Session management
- Chat interaction
- Requirements extraction
- RD generation and export

---

## Definition of Done

- [ ] All 12 acceptance criteria passed and verified
- [ ] Project setup complete with all dependencies installed
- [ ] Layout structure implemented and responsive
- [ ] Session management functional (create, view, switch)
- [ ] Chat interface working with message display
- [ ] WebSocket integration complete with reconnection
- [ ] Requirements panel displaying cards with confidence scores
- [ ] RD preview and Markdown export functional
- [ ] Error handling and offline mode implemented
- [ ] Responsive design working on desktop, tablet, mobile
- [ ] Performance targets met (Lighthouse ≥85)
- [ ] Accessibility compliance (WCAG 2.1 AA, axe audit passing)
- [ ] E2E test passing for complete workflow
- [ ] No console errors or TypeScript warnings
- [ ] Code follows conventions (ESLint, Prettier)
- [ ] All components documented with PropTypes/interfaces

---

## Demo Script for Stakeholders

After completing this story, demonstrate:

```
1. Opening Screen (10 seconds)
   "Here's our Requirements Engineering Platform"
   - Show clean, professional UI
   - Point out session sidebar, chat area, requirements panel

2. Create Session (30 seconds)
   "Let's start a new project"
   - Click "New Session"
   - Type "Mobile Banking App"
   - Session created instantly

3. Conversation (2 minutes)
   "Now I'll gather requirements through conversation"
   - Type: "Users need to transfer money between accounts"
   - Show AI response streaming in
   - Highlight requirement card appearing
   - Point out confidence score (89%)

4. Multiple Requirements (1 minute)
   "Let's add more requirements"
   - Type: "Transactions must complete in under 3 seconds"
   - Second requirement card appears
   - Show different confidence levels

5. RD Generation (1 minute)
   "Now let's generate the document"
   - Click "Generate RD"
   - Show preview loading
   - RD appears formatted
   - Point out traceability to chat

6. Export (30 seconds)
   "Finally, export for the team"
   - Click "Export Markdown"
   - File downloads
   - Open in text editor to show content

Total demo time: ~5 minutes
```

---

## Dependencies for Next Stories

Once Story 5 is complete:

- **STORY-006:** Inference Agent UI (accept/reject inferred requirements)
- **STORY-007:** Validation Agent Visualization (show validation issues)
- **STORY-008:** Inline Editing and Comments
- **STORY-009:** Multi-User Collaboration
- **STORY-010:** PDF Export with Templates

---

## Notes for Windsurf AI Implementation

### Key Implementation Priorities

1. **Start with tooling** - Get Vite + TypeScript working first
2. **Build layout** - AppLayout provides structure for everything
3. **Implement session management** - Foundation for all features
4. **Add chat interface** - Core interaction mechanism
5. **Integrate WebSocket** - Real-time updates critical
6. **Build requirements panel** - Visualization of extracted data
7. **Add RD viewer** - Complete the workflow
8. **Polish with error handling** - Production-ready reliability

### Critical Design Decisions

**Q: Why shadcn/ui over Material-UI or Chakra?**
**A:** Copy-paste components (no npm dependency bloat), full customization, TypeScript-first, accessible by default.

**Q: Why Zustand over Redux?**
**A:** 90% less boilerplate, zero learning curve, <1KB, perfect for MVP.

**Q: Why not use Next.js?**
**A:** MVP doesn't need SSR. SPA with Vite is 10x faster to develop and deploy.

**Q: How to handle large chat histories?**
**A:** Virtual scrolling (react-window) in chat panel, only render visible messages.

### Testing Execution

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Visual regression (optional)
npm run test:visual

# Lighthouse audit
npm run audit

# Accessibility
npm run a11y
```

### Common Pitfalls to Avoid

- ❌ Don't skip TypeScript types - causes bugs later
- ❌ Don't ignore WebSocket reconnection - users will disconnect
- ❌ Don't forget loading states - appears broken without them
- ❌ Don't skip accessibility - legal requirement in many jurisdictions
- ❌ Don't optimize prematurely - get it working first
- ❌ Don't forget responsive design - 40%+ users on mobile
- ❌ Don't skip error boundaries - one error shouldn't crash entire app

### Performance Considerations

- **Code splitting**: Lazy load routes with React.lazy()
- **Memoization**: Use React.memo() for expensive components
- **Debouncing**: Debounce search inputs (300ms)
- **Virtual scrolling**: For chat history > 100 messages
- **Image optimization**: Use WebP format, lazy load
- **Bundle size**: Keep < 200KB gzipped for main bundle

---

## References

- **Design Packet 3**: Frontend Architecture (Section 3-5)
- **WebSocket Protocol**: websocket-protocol.md
- **Vite Configuration**: vite-and-tooling-config.md
- **React Query Docs**: https://tanstack.com/query/latest
- **Zustand Docs**: https://docs.pmnd.rs/zustand
- **shadcn/ui**: https://ui.shadcn.com/
- **Tailwind CSS**: https://tailwindcss.com/
- **WCAG 2.1**: https://www.w3.org/WAI/WCAG21/quickref/

---

**End of Story 5 Document**

**Next Steps:**
1. Set up Vite project with TypeScript
2. Install dependencies (React, Zustand, React Query, shadcn/ui)
3. Create layout structure (AppLayout, Header, Sidebar)
4. Implement session management (store, components, API)
5. Build chat interface (MessageBubble, ChatInput, WebSocket)
6. Add requirements panel (cards, confidence scores)
7. Implement RD viewer and export
8. Add error handling and offline mode
9. Test responsiveness on multiple screen sizes
10. Run Lighthouse and accessibility audits
11. Write E2E tests for complete workflow
12. Polish UI/UX details

**Milestone**: After this story, you have a **fully functional web UI** that users can interact with! 🎉
