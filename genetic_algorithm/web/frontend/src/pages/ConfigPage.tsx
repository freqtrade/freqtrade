import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Settings,
  FileText,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Play,
  Pencil,
  Eye,
  RotateCcw,
} from 'lucide-react';
import { api } from '../api/client';
import type { ConfigTemplate } from '../types';
import { LoadingState } from '../components/StateDisplays';

export function ConfigPage() {
  const navigate = useNavigate();
  const [templates, setTemplates] = useState<ConfigTemplate[]>([]);
  const [original, setOriginal] = useState<Record<string, unknown> | null>(null);
  const [editText, setEditText] = useState('');
  const [selectedName, setSelectedName] = useState('');
  const [editing, setEditing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [validation, setValidation] = useState<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [runId, setRunId] = useState('');
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    api
      .getConfigTemplates()
      .then(setTemplates)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const loadTemplate = async (name: string) => {
    try {
      const config = await api.getConfigTemplate(name);
      setOriginal(config);
      setEditText(JSON.stringify(config, null, 2));
      setSelectedName(name);
      setValidation(null);
      setParseError(null);
      setStartError(null);
      setDirty(false);
      setEditing(false);
    } catch (err) {
      console.error(err);
    }
  };

  const getCurrentConfig = useCallback((): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(editText);
      setParseError(null);
      return parsed;
    } catch (e) {
      setParseError((e as Error).message);
      return null;
    }
  }, [editText]);

  const handleTextChange = (text: string) => {
    setEditText(text);
    setDirty(true);
    setValidation(null);
    // live parse check
    try {
      JSON.parse(text);
      setParseError(null);
    } catch (e) {
      setParseError((e as Error).message);
    }
  };

  const resetToOriginal = () => {
    if (original) {
      setEditText(JSON.stringify(original, null, 2));
      setDirty(false);
      setParseError(null);
      setValidation(null);
    }
  };

  const validateConfig = async () => {
    const config = getCurrentConfig();
    if (!config) return;
    try {
      const result = await api.validateConfig(config);
      setValidation(result);
    } catch (err) {
      console.error(err);
    }
  };

  const startRun = async () => {
    const config = getCurrentConfig();
    if (!config) return;
    setStarting(true);
    setStartError(null);
    try {
      const result = await api.startRun(config, runId || undefined);
      navigate(`/runs/${result.run_id}`);
    } catch (err) {
      setStartError((err as Error).message);
    } finally {
      setStarting(false);
    }
  };

  if (loading) return <LoadingState message="Loading configs..." />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-gray-400" />
          <h1 className="text-2xl font-bold text-gray-100">Configuration</h1>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Template List */}
        <div className="card lg:col-span-1">
          <h2 className="text-sm font-medium text-gray-300 mb-3">Templates</h2>
          <div className="space-y-1.5">
            {templates.map((t) => (
              <button
                key={t.name}
                onClick={() => loadTemplate(t.name)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  selectedName === t.name
                    ? 'bg-accent/10 text-accent border border-accent/20'
                    : 'text-gray-400 hover:bg-white/[0.03] hover:text-gray-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="truncate">{t.name}</span>
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5 ml-6">
                  Pop: {t.population_size} · Gen: {t.generations} · {t.pairs.length} pairs
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Config Editor */}
        <div className="lg:col-span-2 space-y-3">
          {original ? (
            <>
              <div className="card">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <h2 className="text-sm font-medium text-gray-300">{selectedName}</h2>
                    {dirty && (
                      <span className="text-[10px] text-yellow-500 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                        Modified
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {dirty && (
                      <button
                        onClick={resetToOriginal}
                        className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
                        title="Reset to original"
                      >
                        <RotateCcw className="w-3 h-3" /> Reset
                      </button>
                    )}
                    <button
                      onClick={() => setEditing(!editing)}
                      className={`flex items-center gap-1 text-xs transition-colors ${
                        editing
                          ? 'text-accent hover:text-accent/80'
                          : 'text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      {editing ? (
                        <>
                          <Eye className="w-3 h-3" /> Preview
                        </>
                      ) : (
                        <>
                          <Pencil className="w-3 h-3" /> Edit
                        </>
                      )}
                    </button>
                    <button
                      onClick={validateConfig}
                      disabled={!!parseError}
                      className="text-xs text-accent hover:underline disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Validate
                    </button>
                  </div>
                </div>

                {editing ? (
                  <textarea
                    value={editText}
                    onChange={(e) => handleTextChange(e.target.value)}
                    spellCheck={false}
                    className={`w-full text-xs font-mono bg-surface-0 text-gray-300 p-3 rounded-lg border resize-y focus:outline-none focus:ring-1 ${
                      parseError
                        ? 'border-loss/40 focus:ring-loss/40'
                        : 'border-white/5 focus:ring-accent/30'
                    }`}
                    style={{ minHeight: '400px', maxHeight: '700px' }}
                  />
                ) : (
                  <pre className="text-xs text-gray-400 font-mono overflow-x-auto bg-surface-0 p-3 rounded-lg max-h-[500px] overflow-y-auto">
                    {editText}
                  </pre>
                )}

                {parseError && (
                  <div className="flex items-center gap-1.5 mt-2 text-xs text-loss">
                    <XCircle className="w-3 h-3 flex-shrink-0" />
                    JSON parse error: {parseError}
                  </div>
                )}
              </div>

              {/* Validation Results */}
              {validation && (
                <div className="card">
                  <div className="flex items-center gap-2 mb-2">
                    {validation.valid ? (
                      <>
                        <CheckCircle className="w-4 h-4 text-profit" />
                        <span className="text-sm text-profit font-medium">Valid Configuration</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-4 h-4 text-loss" />
                        <span className="text-sm text-loss font-medium">Invalid Configuration</span>
                      </>
                    )}
                  </div>

                  {validation.errors.length > 0 && (
                    <div className="space-y-1 mb-2">
                      {validation.errors.map((e, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-xs text-loss">
                          <XCircle className="w-3 h-3 flex-shrink-0" />
                          {e}
                        </div>
                      ))}
                    </div>
                  )}

                  {validation.warnings.length > 0 && (
                    <div className="space-y-1">
                      {validation.warnings.map((w, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-xs text-warn">
                          <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                          {w}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Start Run Section */}
              <div className="card">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Start Evolution Run</h3>
                <div className="flex items-center gap-3">
                  <input
                    type="text"
                    placeholder="Run ID (optional — auto-generated if blank)"
                    value={runId}
                    onChange={(e) => setRunId(e.target.value)}
                    className="flex-1 bg-surface-0 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-accent/30"
                  />
                  <button
                    onClick={startRun}
                    disabled={starting || !!parseError}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Play className="w-3.5 h-3.5" />
                    {starting ? 'Starting...' : 'Start Run'}
                  </button>
                </div>
                {startError && (
                  <div className="flex items-center gap-1.5 mt-2 text-xs text-loss">
                    <XCircle className="w-3 h-3 flex-shrink-0" />
                    {startError}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="card text-center py-16">
              <Settings className="w-8 h-8 mx-auto mb-2 text-gray-600" />
              <p className="text-gray-500 text-sm">Select a template to view and edit</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
