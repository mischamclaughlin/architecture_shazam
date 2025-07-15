import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';

import HomePage from './pages/HomePage.jsx';
import RegisterPage from './pages/RegisterPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';

import Navbar from './components/Navbar.jsx';

import useLogout from './hooks/useLogout.jsx'


function App() {
  const [user, setUser] = useState(null);
  const logout = useLogout(() => setUser(null));

  useEffect(() => {
    fetch('/api/me', { credentials: 'include' })
      .then(r => r.json())
      .then(js => setUser(js.user))
      .catch(() => setUser(null));
  }, []);

  return (
    <div className='container'>
      <Navbar currentUser={user} onLogout={logout} />

      <Routes>
        <Route path='/' element={<HomePage user={user} />} />
        <Route path='/login' element={<LoginPage onLogin={setUser} />} />
        <Route path='/register' element={<RegisterPage />} />
        <Route path='*' element={<NotFoundPage />} />
      </Routes>
    </div>
  );
}

export default App;