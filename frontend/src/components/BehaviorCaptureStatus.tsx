import { useEffect, useState } from 'react';
import type { BehaviorCollector } from '../collector';

type Activity = { keyboard: boolean; pointer: boolean; interaction: boolean };
type ActivityKey = keyof Activity;

/** Visualizes only observed signal categories, never raw behavioural data. */
export default function BehaviorCaptureStatus({ collector }: { collector: BehaviorCollector }) {
  const [activity, setActivity] = useState<Activity>(() => collector.captureActivity);

  useEffect(() => {
    const update = () => setActivity(collector.captureActivity);
    update();
    const timer = window.setInterval(update, 250);
    return () => window.clearInterval(timer);
  }, [collector]);

  const active = activity.keyboard || activity.pointer || activity.interaction;
  const items: Array<[ActivityKey, string]> = [
    ['keyboard', 'Keystroke rhythm'], ['pointer', 'Pointer dynamics'], ['interaction', 'Interaction pattern'],
  ];
  return <section className={`capture-status ${active ? 'is-capturing' : ''}`} aria-live="polite">
    <div className="capture-status-head"><div><span className="capture-dot" />{active ? 'Capturing behavioural signature' : 'Behavioural protection ready'}</div><span>{active ? 'Observed' : 'Idle'}</span></div>
    <div className="capture-wave" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /></div>
    <div className="capture-signals">{items.map(([key, label]) => <div key={key} className={activity[key] ? 'observed' : ''}><span>{label}</span><b>{activity[key] ? 'Observed' : 'Waiting'}</b></div>)}</div>
    <p>Patterns are captured for verification; this display never shows typed content.</p>
  </section>;
}
