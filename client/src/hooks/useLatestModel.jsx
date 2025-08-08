// ./client/src/hooks/useLatestModel.jsx
import { useState, useEffect, useCallback } from 'react';

export function useLatestModel() {
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(true);

    const reload = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/models', {
                credentials: 'include'
            });
            if (res.status === 404) {
                setModel(null);
            } else if (res.ok) {
                const json = await res.json();
                setModel(json);
            } else {
                console.error('Failed to fetch model:', res.statusText);
                setModel(null);
            }
        } catch (err) {
            console.error('Error fetching model:', err);
            setModel(null);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        reload();
    }, [reload]);

    return { model, loading, reload };
}