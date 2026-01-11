/**
 * Annotation utilities for mapping Freqtrade annotations
 * to TradingView Lightweight Charts markers and price lines
 */

import { ISeriesApi, Time, SeriesMarker } from 'lightweight-charts';

/**
 * Freqtrade annotation types
 */
export interface FreqtradeAreaAnnotation {
  type: 'area';
  start: string; // ISO datetime
  end: string;
  y_start: number;
  y_end: number;
  color?: string;
  label?: string;
  z_level?: number;
}

export interface FreqtradeLineAnnotation {
  type: 'line';
  start: string;
  end: string;
  y_start: number;
  y_end: number;
  color?: string;
  label?: string;
  width?: number;
  line_style?: 'solid' | 'dashed' | 'dotted';
  z_level?: number;
}

export interface FreqtradePointAnnotation {
  type: 'point';
  x: string; // ISO datetime
  y: number;
  color?: string;
  label?: string;
  size?: number;
  shape?: 'circle' | 'rect' | 'roundRect' | 'triangle' | 'pin' | 'arrow' | 'none';
  z_level?: number;
}

export type FreqtradeAnnotation =
  | FreqtradeAreaAnnotation
  | FreqtradeLineAnnotation
  | FreqtradePointAnnotation;

/**
 * Convert ISO datetime string to Unix timestamp for Lightweight Charts
 */
function isoToTime(isoString: string): Time {
  return Math.floor(new Date(isoString).getTime() / 1000) as Time;
}

/**
 * Map Freqtrade line style to Lightweight Charts line style
 */
function mapLineStyle(style?: string): number {
  switch (style) {
    case 'dashed':
      return 2; // LineStyle.Dashed
    case 'dotted':
      return 3; // LineStyle.Dotted
    case 'solid':
    default:
      return 0; // LineStyle.Solid
  }
}

/**
 * Map Freqtrade shape to Lightweight Charts marker shape
 */
function mapMarkerShape(shape?: string): SeriesMarker<Time>['shape'] {
  switch (shape) {
    case 'circle':
      return 'circle';
    case 'rect':
    case 'roundRect':
      return 'square';
    case 'triangle':
      return 'arrowUp';
    case 'pin':
    case 'arrow':
      return 'arrowDown';
    default:
      return 'circle';
  }
}

/**
 * Add area annotation using price lines
 * Note: Lightweight Charts doesn't support filled areas natively,
 * so we use price lines to mark boundaries
 */
export function addAreaAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradeAreaAnnotation
): void {
  const color = annotation.color || 'rgba(255, 193, 7, 0.5)';
  const startTime = isoToTime(annotation.start);
  const endTime = isoToTime(annotation.end);

  // Create upper boundary price line
  series.createPriceLine({
    price: annotation.y_end,
    color: color,
    lineWidth: 1,
    lineStyle: 2, // Dashed
    axisLabelVisible: true,
    title: annotation.label ? `${annotation.label} (upper)` : 'Upper',
  });

  // Create lower boundary price line
  series.createPriceLine({
    price: annotation.y_start,
    color: color,
    lineWidth: 1,
    lineStyle: 2, // Dashed
    axisLabelVisible: true,
    title: annotation.label ? `${annotation.label} (lower)` : 'Lower',
  });
}

/**
 * Add line annotation using price line
 */
export function addLineAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradeLineAnnotation
): void {
  const color = annotation.color || '#ffc107';
  const lineStyle = mapLineStyle(annotation.line_style);
  const lineWidth = annotation.width || 2;

  // For horizontal lines (most common case)
  if (annotation.y_start === annotation.y_end) {
    series.createPriceLine({
      price: annotation.y_start,
      color: color,
      lineWidth: lineWidth,
      lineStyle: lineStyle,
      axisLabelVisible: true,
      title: annotation.label || '',
    });
  } else {
    // For diagonal lines, we need to create multiple price lines
    // or use a separate line series (not implemented here)
    console.warn('Diagonal line annotations are not fully supported. Only horizontal lines are drawn.');

    // Draw at start price as fallback
    series.createPriceLine({
      price: annotation.y_start,
      color: color,
      lineWidth: lineWidth,
      lineStyle: lineStyle,
      axisLabelVisible: true,
      title: annotation.label || '',
    });
  }
}

/**
 * Add point annotation using series marker
 */
export function addPointAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradePointAnnotation,
  existingMarkers: SeriesMarker<Time>[] = []
): SeriesMarker<Time>[] {
  const time = isoToTime(annotation.x);
  const color = annotation.color || '#2196F3';
  const shape = mapMarkerShape(annotation.shape);
  const size = annotation.size || 1;

  const newMarker: SeriesMarker<Time> = {
    time: time,
    position: annotation.y > 0 ? 'aboveBar' : 'belowBar',
    color: color,
    shape: shape,
    text: annotation.label || '',
    size: size,
  };

  // Combine with existing markers and return
  return [...existingMarkers, newMarker];
}

/**
 * Process all annotations and apply them to the chart series
 */
export function applyAnnotations(
  series: ISeriesApi<any>,
  annotations: FreqtradeAnnotation[],
  existingMarkers: SeriesMarker<Time>[] = []
): SeriesMarker<Time>[] {
  let markers = [...existingMarkers];

  annotations.forEach((annotation) => {
    try {
      switch (annotation.type) {
        case 'area':
          addAreaAnnotation(series, annotation);
          break;

        case 'line':
          addLineAnnotation(series, annotation);
          break;

        case 'point':
          markers = addPointAnnotation(series, annotation, markers);
          break;

        default:
          console.warn(`Unknown annotation type:`, annotation);
      }
    } catch (error) {
      console.error(`Failed to add annotation:`, annotation, error);
    }
  });

  // Apply all markers at once
  if (markers.length > 0) {
    series.setMarkers(markers);
  }

  return markers;
}

/**
 * Create a price line helper function
 * Useful for adding support/resistance levels, stop losses, etc.
 */
export function createPriceLine(
  series: ISeriesApi<any>,
  options: {
    price: number;
    color?: string;
    lineWidth?: number;
    lineStyle?: 'solid' | 'dashed' | 'dotted';
    label?: string;
    axisLabelVisible?: boolean;
  }
) {
  return series.createPriceLine({
    price: options.price,
    color: options.color || '#2196F3',
    lineWidth: options.lineWidth || 2,
    lineStyle: mapLineStyle(options.lineStyle),
    axisLabelVisible: options.axisLabelVisible ?? true,
    title: options.label || '',
  });
}

/**
 * Add trade markers to the chart
 * This is useful for marking entry/exit points from trade history
 */
export interface TradeMarker {
  time: Time;
  type: 'entry' | 'exit';
  side: 'long' | 'short';
  price: number;
  label?: string;
}

export function addTradeMarkers(
  series: ISeriesApi<any>,
  trades: TradeMarker[],
  existingMarkers: SeriesMarker<Time>[] = []
): SeriesMarker<Time>[] {
  const markers = [...existingMarkers];

  trades.forEach((trade) => {
    let color: string;
    let shape: SeriesMarker<Time>['shape'];
    let position: SeriesMarker<Time>['position'];
    let text: string;

    if (trade.type === 'entry') {
      if (trade.side === 'long') {
        color = '#26a69a'; // Green
        shape = 'arrowUp';
        position = 'belowBar';
        text = trade.label || 'Buy';
      } else {
        color = '#ef5350'; // Red
        shape = 'arrowDown';
        position = 'aboveBar';
        text = trade.label || 'Short';
      }
    } else {
      if (trade.side === 'long') {
        color = '#ef5350'; // Red
        shape = 'arrowDown';
        position = 'aboveBar';
        text = trade.label || 'Sell';
      } else {
        color = '#26a69a'; // Green
        shape = 'arrowUp';
        position = 'belowBar';
        text = trade.label || 'Cover';
      }
    }

    markers.push({
      time: trade.time,
      position: position,
      color: color,
      shape: shape,
      text: text,
    });
  });

  series.setMarkers(markers);
  return markers;
}

/**
 * Example usage helper that demonstrates how to use annotations
 */
export function exampleAnnotationUsage(
  series: ISeriesApi<any>,
  annotations: FreqtradeAnnotation[]
) {
  // Start with empty markers array
  let markers: SeriesMarker<Time>[] = [];

  // Apply all Freqtrade annotations
  markers = applyAnnotations(series, annotations, markers);

  // Example: Add a custom support level
  createPriceLine(series, {
    price: 50000,
    color: '#4CAF50',
    lineStyle: 'dashed',
    label: 'Support',
  });

  // Example: Add a custom resistance level
  createPriceLine(series, {
    price: 52000,
    color: '#F44336',
    lineStyle: 'dashed',
    label: 'Resistance',
  });

  // Example: Add trade markers
  const tradeMarkers: TradeMarker[] = [
    {
      time: Math.floor(Date.now() / 1000) as Time,
      type: 'entry',
      side: 'long',
      price: 50500,
      label: 'Entry',
    },
  ];

  addTradeMarkers(series, tradeMarkers, markers);
}
