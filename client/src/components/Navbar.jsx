// ./client/src/components/Navbar.jsx
import React from "react";
import { Link } from "react-router-dom";

import './Navbar.css';

export default function Navbar() {
    return (
        <nav className="navbar-menu">
            <Link to="/">
                <h1>Architecture Shazam</h1>
            </Link>
            <Link to="/">Home</Link>
            <Link to="/login">Login</Link>
            <Link to="/register">Register</Link>
        </nav>
    );
}
