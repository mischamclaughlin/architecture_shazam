// ./client/src/hooks/loadMyModels.jsx
import { useState, useEffect, useCallback } from "react";

export function loadMyModels() {
    const [models, setModels] = useState([]);
    const [loadingModels, setLoadingModels] = useState(false);

    const reloadModels = useCallback(async () => {
        setLoadingModels(true);
        const res = await fetch('/api/my_models', { credentials: 'include' });
        const json = await res.json();
        if (res.ok) setModels(json.models || []);
        setLoadingModels(false);
    }, []);

    useEffect(() => { reloadModels(); }, [reloadModels]);

    return { models, loadingModels, reloadModels };
}