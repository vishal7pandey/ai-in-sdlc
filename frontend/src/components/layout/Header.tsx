import { Menu } from 'lucide-react'

interface HeaderProps {
  onToggleSidebar?: () => void
}

export function Header({ onToggleSidebar }: HeaderProps) {
  return (
    <header className="h-14 border-b border-slate-800 flex items-center justify-between px-4 bg-slate-950/90 backdrop-blur">
      <div className="flex items-center gap-2">
        {onToggleSidebar && (
          <button
            type="button"
            onClick={onToggleSidebar}
            className="mr-2 p-1 rounded-md hover:bg-slate-800 md:hidden"
            aria-label="Toggle session sidebar"
          >
            <Menu className="h-4 w-4" />
          </button>
        )}
        <span className="font-semibold text-slate-100">Requirements Engineering Platform</span>
        <span className="text-xs text-slate-400">MVP</span>
      </div>
      <div className="text-xs text-slate-400">User Menu</div>
    </header>
  )
}
