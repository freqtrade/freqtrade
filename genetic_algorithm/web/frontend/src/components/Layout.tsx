import { Outlet, useLocation, Link } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { ToastContainer } from './Toast';
import { ErrorBoundary } from './ErrorBoundary';
import { useStore } from '../store/useStore';
import { ChevronRight, WifiOff } from 'lucide-react';

function Breadcrumbs() {
  const location = useLocation();
  const parts = location.pathname.split('/').filter(Boolean);

  if (parts.length === 0) return null;

  const crumbs: { label: string; path: string }[] = [];
  let path = '';
  for (const p of parts) {
    path += `/${p}`;
    crumbs.push({ label: p.replace(/-/g, ' '), path });
  }

  return (
    <nav className="flex items-center gap-1 text-xs text-gray-500 mb-4">
      <Link to="/" className="hover:text-gray-300 transition-colors">Home</Link>
      {crumbs.map((c, i) => (
        <span key={c.path} className="flex items-center gap-1">
          <ChevronRight className="w-3 h-3" />
          {i === crumbs.length - 1 ? (
            <span className="text-gray-300 capitalize">{c.label}</span>
          ) : (
            <Link to={c.path} className="hover:text-gray-300 transition-colors capitalize">
              {c.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}

export function Layout() {
  const connected = useStore((s) => s.connected);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-6 overflow-auto">
        {/* Reconnection banner */}
        {!connected && (
          <div className="mb-4 flex items-center gap-2 px-4 py-2.5 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-xs">
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            <span>Connection lost — reconnecting automatically...</span>
            <div className="ml-auto w-3 h-3 border-2 border-yellow-400/60 border-t-yellow-400 rounded-full animate-spin" />
          </div>
        )}
        <Breadcrumbs />
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
      <ToastContainer />
    </div>
  );
}
