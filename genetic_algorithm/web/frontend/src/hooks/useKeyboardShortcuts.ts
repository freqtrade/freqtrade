/**
 * Keyboard shortcuts hook — provides vim-style "g" prefix navigation
 * and common shortcuts like Esc to close modals.
 *
 * Shortcuts:
 *   g h → Home (Dashboard)
 *   g r → Runs
 *   g f → Hall of Fame
 *   g c → Config
 *   g a → Analytics
 *   Esc → close modals / go back
 *   ?   → show help (future)
 */

import { useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

const SEQUENCE_TIMEOUT = 800; // ms to wait for second key in sequence

export function useKeyboardShortcuts() {
  const navigate = useNavigate();
  const pendingPrefix = useRef<string | null>(null);
  const prefixTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs/textareas
    const target = e.target as HTMLElement;
    if (
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.tagName === 'SELECT' ||
      target.isContentEditable
    ) {
      // Still handle Esc in inputs
      if (e.key === 'Escape') {
        (target as HTMLInputElement).blur();
      }
      return;
    }

    // Esc — close modals, details, go back
    if (e.key === 'Escape') {
      // Close any open <details> element
      const openDetails = document.querySelector('details[open]');
      if (openDetails) {
        (openDetails as HTMLDetailsElement).open = false;
        return;
      }
      return;
    }

    // "g" prefix — start sequence
    if (e.key === 'g' && !pendingPrefix.current) {
      pendingPrefix.current = 'g';
      clearTimeout(prefixTimer.current);
      prefixTimer.current = setTimeout(() => {
        pendingPrefix.current = null;
      }, SEQUENCE_TIMEOUT);
      return;
    }

    // Handle second key in "g" sequence
    if (pendingPrefix.current === 'g') {
      pendingPrefix.current = null;
      clearTimeout(prefixTimer.current);

      switch (e.key) {
        case 'h':
          navigate('/');
          break;
        case 'r':
          navigate('/runs');
          break;
        case 'f':
          navigate('/hall-of-fame');
          break;
        case 'c':
          navigate('/config');
          break;
        case 'a':
          navigate('/analytics');
          break;
      }
    }
  }, [navigate]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearTimeout(prefixTimer.current);
    };
  }, [handleKeyDown]);
}
