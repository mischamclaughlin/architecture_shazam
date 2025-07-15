// ./client/src/components/Navbar.jsx
import React from "react";
import { Link } from "react-router-dom";

import './Navbar.css';

export default function Navbar({ currentUser, onLogout }) {
    return (
        <nav className="navbar-menu">
            <Link to="/">
                <h1 className="nav-title">Architecture Shazam</h1>
            </Link>
            {currentUser ? (
                <div className="nav-area">
                    <p className="welcome-user">Welcome, {currentUser.username}</p>
                    <Link to="/">Home</Link>
                    <button onClick={onLogout} className="logout-button">
                        Logout
                    </button>
                </div>
            ) : (
                <div className="nav-area">
                    <Link to="/">Home</Link>
                    <Link to="/login">Login</Link>
                    <Link to="/register">Register</Link>
                </div>
            )}
        </nav >
    );
}