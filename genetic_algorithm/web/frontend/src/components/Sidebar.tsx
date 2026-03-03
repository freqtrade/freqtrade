import { NavLink, useLocation } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Trophy,
  Settings,
  Dna,
  Wifi,
  WifiOff,
  GitCompare,
  Brain,
  Layers,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useStore } from '../store/useStore';
import { ThemeToggle } from './ThemeToggle';

const navItems = [
  { to: '/',              icon: Activity,   label: 'Dashboard' },
  { to: '/runs',          icon: Dna,        label: 'Runs' },
  { to: '/hall-of-fame',  icon: Trophy,     label: 'Hall of Fame' },
  { to: '/compare',       icon: GitCompare, label: 'Compare' },
  { to: '/runs/compare',  icon: Layers,     label: 'Run Compare' },
  { to: '/analytics',     icon: Brain,      label: 'Analytics' },
  { to: '/config',        icon: Settings,   label: 'Config' },
];

export function Sidebar() {
  const connected = useStore((s) => s.connected);
  const runsMap = useStore((s) => s.runs);
  const activeRuns = Array.from(runsMap.values()).filter(
    (r) => r.status === 'running' || r.status === 'paused',
  ).length;

  return (
    <aside className="w-56 bg-surface-1 border-r border-white/5 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 py-4 border-b border-white/5">
        <BarChart3 className="w-6 h-6 text-accent" />
        <span className="font-semibold text-sm tracking-tight">GA Dashboard</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors',
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-white/[0.03]',
              )
            }
          >
            <Icon className="w-4 h-4" />
            {label}
            {label === 'Runs' && activeRuns > 0 && (
              <span className="ml-auto bg-accent/20 text-accent text-[10px] font-semibold px-1.5 py-0.5 rounded-full">
                {activeRuns}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Connection status + Theme toggle */}
      <div className="px-4 py-3 border-t border-white/5 flex items-center gap-2 text-xs">
        {connected ? (
          <>
            <Wifi className="w-3.5 h-3.5 text-profit" />
            <span className="text-gray-400">Connected</span>
          </>
        ) : (
          <>
            <WifiOff className="w-3.5 h-3.5 text-loss" />
            <span className="text-gray-500">Disconnected</span>
          </>
        )}
        <span className="ml-auto" />
        <ThemeToggle />
      </div>
    </aside>
  );
}
