// ./client/src/hooks/useDeleteImage.jsx
import { useCallback } from 'react';

export function useDeleteImage() {
    return useCallback(async (id) => {
        if (!id) throw new Error('No image id provided');
        const res = await fetch(`/api/images/${id}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to delete image');
        }
        return res.json();
    }, []);
}