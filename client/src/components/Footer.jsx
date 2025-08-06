// ./client/src/components/Footer.jsx
import React from "react";
import { Link } from "react-router-dom";

import './Footer.css';

export default function Navbar({ currentUser, onLogout }) {
    return (
        <footer className="footer-menu">
            <p>©ArchitectureShazam</p>
        </footer>
    );
}
