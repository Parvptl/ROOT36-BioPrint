import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import AnimatedSecurityBackground from './components/AnimatedSecurityBackground';
import EnrollPage from './pages/EnrollPage';
import LoginPage from './pages/LoginPage';

const Icon = ({ children }: { children: React.ReactNode }) => <span className="nav-icon" aria-hidden="true">{children}</span>;
export default function App() {
  const location = useLocation();
  const isLogin = location.pathname === '/login' || location.pathname === '/';
  return <div className="app-shell video-shell">
    <AnimatedSecurityBackground login={isLogin} />
    {!isLogin && <aside className="sidebar">
      <NavLink className="brand" to="/login" aria-label="BioPrint home"><div className="brand-mark">⌁</div><div><div className="brand-name">BioPrint</div><div className="brand-tag">Identity intelligence</div></div></NavLink>
      <div className="nav-label">Workspace</div>
      <nav className="navlinks" aria-label="Primary navigation">
        <NavLink to="/login"><Icon>⌁</Icon>Secure access</NavLink>
        <NavLink to="/enroll"><Icon>◌</Icon>Identity enrollment</NavLink>

      </nav>
      <div className="sidebar-foot"><span className="status-dot" />Behavioural engine ready</div>
    </aside>}
    <main className={isLogin ? 'main-area main-area--login' : 'main-area'}>{!isLogin && <header className="topbar"><div className="crumb"><span>BioPrint</span><b>/</b><span>Behavioural authentication</span></div><div className="system-status"><span className="status-dot" />System protected</div></header>}
      <div className="page-content"><Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/enroll" element={<EnrollPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes></div>
    </main>
  </div>;
}
