// ./client/src/pages/LoginPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import './LoginPage.css'

export default function LoginPage({ onLogin }) {
    const [form, setForm] = useState({
        username: '',
        password: '',
    });
    const [error, setError] = useState('');
    const [status, setStatus] = useState('');
    const navigate = useNavigate();

    const handleChange = e => {
        const { name, value } = e.target;
        setForm(f => ({ ...f, [name]: value }));
    };

    const handleSubmit = async e => {
        e.preventDefault();
        setError('');
        setStatus('');

        if (!form.username || !form.password) {
            setError('All fields are required');
            return;
        }

        setStatus('Logging in...');
        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: form.username,
                    password: form.password,
                }),
            });
            const json = await res.json();
            if (!res.ok) throw new Error(json.error || res.statusText);

            onLogin(json.user)
            navigate('/')

            setStatus('Login successful!');
            setForm({ username: '', password: '' });
        } catch (err) {
            console.log(err);
            setError(err.message);
            setStatus('');
        }
    };

    return (
        <div className='login-page'>
            <h2>Log In</h2>
            <form className='login-form' onSubmit={handleSubmit}>
                {error && <p className='form-error'>{error}</p>}
                <label>
                    Username
                    <input
                        type='text'
                        name='username'
                        value={form.username}
                        onChange={handleChange}
                    />
                </label>

                <label>
                    Password
                    <input
                        type='password'
                        name='password'
                        value={form.password}
                        onChange={handleChange}
                    />
                </label>

                <button type='submit' disabled={status === 'Logging in...'}>
                    {status || 'Login'}
                </button>
            </form>
        </div>
    );
}