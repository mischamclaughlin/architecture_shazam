// ./client/src/App.jsx
import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';

import HomePage from './pages/HomePage.jsx';
import RegisterPage from './pages/RegisterPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';
import GalleryPage from './pages/GalleryPage.jsx';

import Navbar from './components/Navbar.jsx';
import Footer from './components/Footer.jsx';

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

      <main className='content'>
        <Routes>
          <Route path='/' element={<HomePage user={user} />} />
          <Route path='/gallery' element={<GalleryPage user={user} />} />
          <Route path='/login' element={<LoginPage onLogin={setUser} />} />
          <Route path='/register' element={<RegisterPage />} />
          <Route path='*' element={<NotFoundPage />} />
        </Routes>
      </main>

      <Footer />
    </div>
  );
}

export default App;