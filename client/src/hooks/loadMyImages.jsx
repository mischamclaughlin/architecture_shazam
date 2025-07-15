// ./client/src/hooks/loadMyImages.jsx
import { useState, useEffect, useCallback } from "react";

export function loadMyImages() {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(false);

    const reload = useCallback(async () => {
        setLoading(true);
        const res = await fetch('/api/my_images', { credentials: 'include' });
        const json = await res.json();
        if (res.ok) setImages(json.images);
        setLoading(false);
    }, []);

    useEffect(() => {
        reload();
    }, [reload]);

    return { images, loading, reload };
}