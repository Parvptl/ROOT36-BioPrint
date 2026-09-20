import type { ReactNode } from 'react';

type PageFrameProps = {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
};

/** Shared internal-page hierarchy for BioPrint's security workflows. */
export default function PageFrame({ eyebrow, title, description, children }: PageFrameProps) {
  return <section className="page-frame">
    <header className="page-frame-head">
      <div className="console-kicker">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{description}</p>
    </header>
    <div className="page-frame-body">{children}</div>
  </section>;
}
