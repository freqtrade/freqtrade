import type { StrategyGene } from '../types';
import { clsx } from 'clsx';

interface StrategyGeneTreeProps {
  gene: StrategyGene;
}

export function StrategyGeneTree({ gene }: StrategyGeneTreeProps) {
  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-medium text-gray-300">Strategy Gene</h3>

      {/* Parameters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <ParamPill label="Timeframe" value={gene.timeframe} />
        <ParamPill label="Stoploss" value={`${(gene.stoploss * 100).toFixed(1)}%`} negative />
        <ParamPill label="Max Trades" value={gene.max_open_trades.toString()} />
        <ParamPill label="Trailing" value={gene.trailing_stop ? 'Yes' : 'No'} />
      </div>

      {/* Indicators */}
      <Section title={`Indicators (${gene.indicators.length})`}>
        <div className="space-y-1.5">
          {gene.indicators.map((ind, i) => (
            <div key={i} className="flex items-center gap-2 text-xs px-2 py-1.5 bg-surface-2 rounded-lg">
              <span className="text-accent font-mono font-medium">{ind.type}</span>
              <span className="text-gray-500">
                {Object.entries(ind.parameters)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(', ')}
              </span>
              {ind.timeframe && (
                <span className="ml-auto text-gray-600 text-[10px]">{ind.timeframe}</span>
              )}
            </div>
          ))}
        </div>
      </Section>

      {/* Entry Conditions */}
      <Section title={`Entry Conditions (${gene.entry_conditions.length})`}>
        <ConditionList conditions={gene.entry_conditions} color="text-profit" />
      </Section>

      {/* Exit Conditions */}
      <Section title={`Exit Conditions (${gene.exit_conditions.length})`}>
        <ConditionList conditions={gene.exit_conditions} color="text-loss" />
      </Section>

      {/* ROI Table */}
      {Object.keys(gene.minimal_roi).length > 0 && (
        <Section title="Minimal ROI">
          <div className="flex flex-wrap gap-2">
            {Object.entries(gene.minimal_roi)
              .sort(([a], [b]) => parseInt(a) - parseInt(b))
              .map(([mins, pct]) => (
                <span key={mins} className="text-xs bg-surface-2 px-2 py-1 rounded font-mono">
                  {mins}m → {(pct * 100).toFixed(1)}%
                </span>
              ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1.5">{title}</h4>
      {children}
    </div>
  );
}

function ParamPill({
  label,
  value,
  negative,
}: {
  label: string;
  value: string;
  negative?: boolean;
}) {
  return (
    <div className="bg-surface-2 rounded-lg px-3 py-2">
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={clsx('text-sm font-mono font-medium', negative ? 'text-loss' : 'text-gray-200')}>
        {value}
      </div>
    </div>
  );
}

function ConditionList({
  conditions,
  color,
}: {
  conditions: { indicator: string; operator: string; threshold: unknown; logic: string }[];
  color: string;
}) {
  return (
    <div className="space-y-1">
      {conditions.map((c, i) => (
        <div key={i} className="flex items-center gap-1.5 text-xs">
          {i > 0 && (
            <span className="text-gray-600 font-mono text-[10px] w-6">{c.logic}</span>
          )}
          <span className={clsx('font-mono font-medium', color)}>{c.indicator}</span>
          <span className="text-gray-400">{c.operator}</span>
          <span className="text-gray-300 font-mono">{String(c.threshold ?? '')}</span>
        </div>
      ))}
    </div>
  );
}
