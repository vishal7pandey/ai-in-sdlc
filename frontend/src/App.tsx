import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import { AppLayout } from './components/layout/AppLayout'

const HomePage = lazy(() => import('./pages/HomePage').then((m) => ({ default: m.HomePage })))
const SessionPage = lazy(() => import('./pages/SessionPage').then((m) => ({ default: m.SessionPage })))

function App() {
  return (
    <BrowserRouter>
      <Suspense
        fallback={
          <div className="h-screen flex items-center justify-center text-sm text-slate-500 bg-slate-950">
            Loading application...
          </div>
        }
      >
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/sessions/:sessionId" element={<SessionPage />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}

export default App
