import { useEffect, useRef, useState } from 'react';

const LOGIN_VIDEO_URL = 'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260702_051048_5ef213b5-26db-4da8-b604-7ef823760b6b.mp4';
const INTERNAL_VIDEO_URL = 'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260613_180732_a54afbf6-b30d-470e-861f-669871f09f67.mp4';

/** A single persistent atmospheric layer. It never participates in page layout. */
export default function AnimatedSecurityBackground({ login }: { login: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const layer = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(true);
  const source = login ? LOGIN_VIDEO_URL : INTERNAL_VIDEO_URL;

  useEffect(() => {
    const syncPlayback = () => {
      if (!video.current) return;
      if (document.hidden) video.current.pause();
      else void video.current.play().catch(() => undefined);
    };
    document.addEventListener('visibilitychange', syncPlayback);
    syncPlayback();
    return () => document.removeEventListener('visibilitychange', syncPlayback);
  }, []);

  useEffect(() => {
    setAvailable(true);
  }, [source]);

  useEffect(() => {
    if (login || !window.matchMedia('(pointer: fine)').matches) return;
    let frame = 0;
    const moveAmbient = (event: PointerEvent) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        layer.current?.style.setProperty('--ambient-x', `${(event.clientX / window.innerWidth) * 100}%`);
        layer.current?.style.setProperty('--ambient-y', `${(event.clientY / window.innerHeight) * 100}%`);
      });
    };
    window.addEventListener('pointermove', moveAmbient, { passive: true });
    return () => { window.removeEventListener('pointermove', moveAmbient); cancelAnimationFrame(frame); };
  }, [login]);

  return <div ref={layer} className={`cyber-background ${login ? 'cyber-background--login' : 'cyber-background--internal'} ${available ? '' : 'is-fallback'}`} aria-hidden="true">
    {available ? <video key={source} ref={video} autoPlay muted loop playsInline preload="metadata" onError={() => setAvailable(false)}>
      <source src={source} type="video/mp4" />
    </video> : null}
    <div className="cyber-background-scrim" />
    <div className="cyber-background-ambient" />
  </div>;
}
