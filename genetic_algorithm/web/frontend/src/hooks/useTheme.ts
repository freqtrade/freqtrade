/**
 * useTheme — Dark/light theme toggle with localStorage persistence.
 *
 * Uses Tailwind's `class` strategy.  The <html> element gets
 * class="dark" (default) or class="light".
 */

import { useEffect, useState, useCallback } from 'react';

type Theme = 'dark' | 'light';

const STORAGE_KEY = 'ga-dashboard-theme';

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'dark';
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === 'dark' || stored === 'light') return stored;
  return 'dark'; // default
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  // Apply class to <html>
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('dark', 'light');
    root.classList.add(theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setThemeState((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
  }, []);

  return { theme, toggle, setTheme };
}
