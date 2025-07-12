// ./client/src/pages/RegisterPage.jsx
import React, { useState } from 'react';

import './RegisterPage.css'

export default function RegisterPage() {
    const [form, setForm] = useState({
        username: '',
        email: '',
        password: '',
        confirm: '',
    });
    const [error, setError] = useState('');
    const [status, setStatus] = useState('');

    const handleChange = e => {
        const { name, value } = e.target;
        setForm(f => ({ ...f, [name]: value }));
    };

    const handleSubmit = async e => {
        e.preventDefault();
        setError('');
        setStatus('');

        if (!form.username || !form.email || !form.password) {
            setError('All fields are required');
            return;
        }
        if (form.password !== form.confirm) {
            setError('Password do not match');
            return;
        }

        setStatus('Registering...');
        try {
            const res = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: form.username,
                    email: form.email,
                    password: form.password,
                }),
            });
            const json = await res.json();
            if (!res.ok) throw new Error(json.error || res.statusText);
            setStatus('Registration successful! You can now log in');
            setForm({ username: '', email: '', password: '', confirm: '' });
        } catch (err) {
            console.log(err);
            setError(err.message);
            setStatus('');
        }
    };

    return (
        <div className='register-page'>
            <h2>Create Account</h2>
            <form className='register-form' onSubmit={handleSubmit}>
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
                    Email
                    <input
                        type='email'
                        name='email'
                        value={form.email}
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

                <label>
                    Confirm Password
                    <input
                        type='password'
                        name='confirm'
                        value={form.confirm}
                        onChange={handleChange}
                    />
                </label>

                <button type='submit' disabled={status === 'Registering...'}>
                    {status || 'Register'}
                </button>
            </form>
        </div>
    );
}