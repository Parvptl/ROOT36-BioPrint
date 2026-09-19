import { NavLink, Navigate, Route, Routes } from 'react-router-dom';

import EnrollPage from './pages/EnrollPage';
import LoginPage from './pages/LoginPage';

export default function App() {
  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <div className="brand-mark">B</div>
          <div>
            <div className="brand-name">BioPrint</div>
            <div className="brand-tag">Behavioural Authentication</div>
          </div>
        </div>
        <nav className="navlinks">
          <NavLink to="/login">Login</NavLink>
          <NavLink to="/enroll">Enroll</NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/enroll" element={<EnrollPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </div>
  );
}
