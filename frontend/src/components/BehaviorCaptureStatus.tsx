import { useEffect, useState } from 'react';
import type { BehaviorCollector } from '../collector';

type ActivityKey = 'keyboard' | 'pointer' | 'interaction';
type Activity = Record<ActivityKey, number>;

/** Visualizes privacy-safe event counts, never raw behavioural data. */
export default function BehaviorCaptureStatus({ collector }: { collector: BehaviorCollector }) {
  const [activity, setActivity] = useState<Activity>(() => collector.captureCounts);

  useEffect(() => {
    const update = () => setActivity(collector.captureCounts);
    update();
    const timer = window.setInterval(update, 250);
    return () => window.clearInterval(timer);
  }, [collector]);

  const active = activity.keyboard > 0 || activity.pointer > 0 || activity.interaction > 0;
  const items: Array<[ActivityKey, string]> = [
    ['keyboard', 'Keystroke rhythm'], ['pointer', 'Pointer dynamics'], ['interaction', 'Interaction pattern'],
  ];
  return <section className={`capture-status ${active ? 'is-capturing' : ''}`} aria-live="polite">
    <div className="capture-status-head"><div><span className="capture-dot" />{active ? 'Capturing behavioural signature' : 'Behavioural protection ready'}</div><span>{active ? 'Observed' : 'Idle'}</span></div>
    <div className="capture-wave" aria-hidden="true">{Array.from({ length: 8 }, (_, index) => <i key={index} style={{ height: `${Math.min(20, 3 + (activity.keyboard + activity.pointer + index) % 18)}px` }} />)}</div>
    <div className="capture-signals">{items.map(([key, label]) => <div key={key} className={activity[key] > 0 ? 'observed' : ''}><span>{label}</span><b>{activity[key] > 0 ? `${activity[key]} events` : 'Waiting'}</b></div>)}</div>
    <p>Patterns are captured for verification; this display never shows typed content.</p>
  </section>;
}
