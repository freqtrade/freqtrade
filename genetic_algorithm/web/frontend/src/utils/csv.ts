/**
 * CSV export utility — converts array of objects to CSV and triggers download.
 */

export function exportToCsv(
  filename: string,
  data: Record<string, unknown>[],
  columns?: { key: string; label?: string }[],
) {
  if (data.length === 0) return;

  // Determine columns: either explicit or auto-detect from first row
  const cols = columns ?? Object.keys(data[0]).map((k) => ({ key: k, label: k }));

  // Header row
  const header = cols.map((c) => escapeCsv(c.label ?? c.key)).join(',');

  // Data rows
  const rows = data.map((row) =>
    cols
      .map((c) => {
        const val = row[c.key];
        if (val === null || val === undefined) return '';
        if (typeof val === 'number') return String(val);
        return escapeCsv(String(val));
      })
      .join(','),
  );

  const csv = [header, ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.csv') ? filename : `${filename}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeCsv(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}
