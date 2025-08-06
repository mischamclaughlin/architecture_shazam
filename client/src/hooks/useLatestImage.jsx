// ./client/src/hooks/useLatestImage.jsx
import { useState, useEffect, useCallback } from 'react';

export function useLatestImage() {
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(true);

    const reload = useCallback(async () => {
        setLoading(true);
        const res = await fetch('/api/image', { credentials: 'include' });
        const json = await res.json();
        if (res.ok) setImage(json);
        else setImage(null);
        setLoading(false);
    }, []);

    useEffect(() => {
        reload();
    }, [reload]);

    return { image, loading, reload };
}