// ./client/src/hooks/logout.jsx
import { useCallback } from "react";
import { useNavigate } from "react-router-dom";

export default function useLogout(onClientClear) {
    const navigate = useNavigate()

    return useCallback(async () => {
        try {
            const res = await fetch('/api/logout', { method: 'POST' })
            if (!res.ok) throw new Error('Logout failed');

            onClientClear?.();

            navigate('/login');
        } catch (err) {
            console.error('Logout error:', err);
            alert('Could not log out. Please try again.');
        }
    }, [navigate, onClientClear]);
}