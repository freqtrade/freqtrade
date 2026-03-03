import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { X, Play, CheckCircle, XCircle, AlertTriangle, FileText } from 'lucide-react';
import { api } from '../api/client';
import type { ConfigTemplate } from '../types';

interface StartRunDialogProps {
  open: boolean;
  onClose: () => void;
}

export function StartRunDialog({ open, onClose }: StartRunDialogProps) {
  const navigate = useNavigate();
  const [templates, setTemplates] = useState<ConfigTemplate[]>([]);
  const [selectedName, setSelectedName] = useState('');
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [configText, setConfigText] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [runId, setRunId] = useState('');
  const [validation, setValidation] = useState<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  } | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      api.getConfigTemplates().then(setTemplates).catch(console.error);
      setError(null);
      setValidation(null);
      setStarting(false);
    }
  }, [open]);

  const loadTemplate = async (name: string) => {
    try {
      const cfg = await api.getConfigTemplate(name);
      setSelectedName(name);
      setConfig(cfg);
      setConfigText(JSON.stringify(cfg, null, 2));
      setValidation(null);
      setEditMode(false);
      setError(null);
    } catch (err) {
      console.error(err);
    }
  };

  const handleValidate = async () => {
    let cfgToValidate = config;
    if (editMode) {
      try {
        cfgToValidate = JSON.parse(configText);
        setConfig(cfgToValidate);
      } catch {
        setValidation({ valid: false, errors: ['Invalid JSON syntax'], warnings: [] });
        return;
      }
    }
    if (!cfgToValidate) return;
    try {
      const result = await api.validateConfig(cfgToValidate);
      setValidation(result);
    } catch (err) {
      setError(String(err));
    }
  };

  const handleStart = async () => {
    let cfgToStart = config;
    if (editMode) {
      try {
        cfgToStart = JSON.parse(configText);
      } catch {
        setError('Invalid JSON — fix syntax before starting');
        return;
      }
    }
    if (!cfgToStart) {
      setError('Select a config template first');
      return;
    }

    setStarting(true);
    setError(null);
    try {
      const run = await api.startRun(cfgToStart, runId || undefined);
      onClose();
      navigate(`/runs/${run.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStarting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Dialog */}
      <div className="relative bg-surface-1 border border-white/10 rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
          <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
            <Play className="w-4 h-4 text-profit" /> Start New Evolution
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Template Selector */}
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wider block mb-2">
              Config Template
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {templates.map((t) => (
                <button
                  key={t.name}
                  onClick={() => loadTemplate(t.name)}
                  className={`text-left px-3 py-2.5 rounded-lg text-sm transition-colors border ${
                    selectedName === t.name
                      ? 'bg-accent/10 text-accent border-accent/30'
                      : 'text-gray-400 hover:bg-white/[0.03] hover:text-gray-200 border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate font-medium">{t.name}</span>
                  </div>
                  <div className="text-[10px] text-gray-500 mt-0.5 ml-6">
                    Pop: {t.population_size} · Gen: {t.generations} · {t.pairs.length} pairs
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Run ID (optional) */}
          <div>
            <label className="text-xs text-gray-400 uppercase tracking-wider block mb-1.5">
              Run ID <span className="normal-case text-gray-600">(optional — auto-generated if blank)</span>
            </label>
            <input
              type="text"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              placeholder="e.g. experiment-v2"
              className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-accent/50"
            />
          </div>

          {/* Config Preview / Editor */}
          {config && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-gray-400 uppercase tracking-wider">
                  Configuration
                </label>
                <button
                  onClick={() => {
                    if (!editMode) setConfigText(JSON.stringify(config, null, 2));
                    setEditMode(!editMode);
                  }}
                  className="text-xs text-accent hover:underline"
                >
                  {editMode ? 'Preview' : 'Edit JSON'}
                </button>
              </div>
              {editMode ? (
                <textarea
                  value={configText}
                  onChange={(e) => {
                    setConfigText(e.target.value);
                    setValidation(null);
                  }}
                  className="w-full bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-xs text-gray-300 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50 resize-y"
                  rows={16}
                  spellCheck={false}
                />
              ) : (
                <pre className="text-xs text-gray-400 font-mono overflow-x-auto bg-surface-0 p-3 rounded-lg max-h-[300px] overflow-y-auto border border-white/5">
                  {JSON.stringify(config, null, 2)}
                </pre>
              )}
            </div>
          )}

          {/* Validation Results */}
          {validation && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                {validation.valid ? (
                  <>
                    <CheckCircle className="w-4 h-4 text-profit" />
                    <span className="text-sm text-profit font-medium">Config is valid</span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-4 h-4 text-loss" />
                    <span className="text-sm text-loss font-medium">Invalid config</span>
                  </>
                )}
              </div>
              {validation.errors.map((e, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs text-loss ml-6">
                  <XCircle className="w-3 h-3 flex-shrink-0 mt-0.5" /> {e}
                </div>
              ))}
              {validation.warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs text-warn ml-6">
                  <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" /> {w}
                </div>
              ))}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 text-sm text-loss bg-loss/10 px-3 py-2 rounded-lg">
              <XCircle className="w-4 h-4 flex-shrink-0" /> {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-5 py-4 border-t border-white/5">
          <button
            onClick={handleValidate}
            disabled={!config}
            className="text-sm text-gray-400 hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2"
          >
            Validate
          </button>
          <button
            onClick={onClose}
            className="text-sm text-gray-400 hover:text-gray-200 transition-colors px-4 py-2"
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={!config || starting}
            className="flex items-center gap-2 bg-profit hover:bg-profit/80 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2 rounded-lg transition-colors"
          >
            <Play className="w-3.5 h-3.5" />
            {starting ? 'Starting...' : 'Start Evolution'}
          </button>
        </div>
      </div>
    </div>
  );
}
