import React, { useState } from 'react';
import { Routes, Route, Link } from 'react-router-dom';

import HomePage from './pages/HomePage.jsx';
import RegisterPage from './pages/RegisterPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';

import Navbar from './components/Navbar.jsx';



function App() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path='/' element={<HomePage />} />
        <Route path='/login' element={<LoginPage />} />
        <Route path='/register' element={<RegisterPage />} />
        <Route path='*' element={<NotFoundPage />} />
      </Routes>
    </>
  );
}

export default App;