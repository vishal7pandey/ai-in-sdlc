import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Header } from './Header'
import { SessionSidebar } from '../session/SessionSidebar'

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100">
      <Header onToggleSidebar={() => setSidebarOpen((open) => !open)} />
      <div className="flex flex-1 overflow-hidden">
        <div
          className={
            'transition-transform absolute md:relative z-20 md:z-0 ' +
            (sidebarOpen ? 'translate-x-0 md:translate-x-0' : '-translate-x-full md:translate-x-0')
          }
        >
          <SessionSidebar />
        </div>
        <main className="flex-1 overflow-auto bg-slate-900 border-l border-slate-800">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
