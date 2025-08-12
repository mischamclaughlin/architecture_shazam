// ./client/src/components/Navbar.jsx
import React from "react";
import { Link } from "react-router-dom";
import logo from '../assets/logo.png'

import './Navbar.css';

export default function Navbar({ currentUser, onLogout }) {
    return (
        <nav className="navbar-menu">
            <Link to="/">
                <img src={logo} alt="Architecture Shazam logo" className="brand-logo" />
            </Link>

            {currentUser ? (
                <div className="nav-area">
                    <p className="welcome-user">Welcome, {currentUser.username}</p>

                    <div className="dropdown">
                        <button className="dropbtn">Menu ↴</button>
                        <ul className="dropdown-content">
                            <li><Link to="/">Home</Link></li>
                            <li><Link to="/gallery">Gallery</Link></li>
                            <li>
                                <button onClick={onLogout} className="logout-button">
                                    Logout
                                </button>
                            </li>
                        </ul>
                    </div>
                </div>
            ) : (
                <div className="nav-area">
                    <Link to="/">Home</Link>
                    <Link to="/login">Login</Link>
                    <Link to="/register">Register</Link>
                </div>
            )}
        </nav>
    );
}