import { useCallback, useEffect, useRef } from 'react';

import { BehaviorCollector } from '../collector';
import type { FieldContext } from '../collector';

/**
 * Owns a BehaviorCollector for the lifetime of a component.
 *
 * The collector lives in a ref rather than state: it mutates on every
 * keystroke and pointer move, and putting that in state would re-render the
 * page hundreds of times per capture — which would itself perturb the timing
 * we are trying to measure.
 */
export function useBehaviorCollector() {
  const collectorRef = useRef<BehaviorCollector | null>(null);
  if (collectorRef.current === null) {
    collectorRef.current = new BehaviorCollector();
  }
  const collector = collectorRef.current;

  useEffect(() => {
    collector.start();
    return () => collector.stop();
  }, [collector]);

  /**
   * Ref callback that registers a field with the collector.
   *
   * Returned memoised per context so React does not detach and reattach the
   * ref on every render.
   */
  const bind = useCallback(
    (ctx: FieldContext) => (element: HTMLInputElement | null) => {
      if (element) collector.observeField(element, ctx);
    },
    [collector],
  );

  return { collector, bind };
}
