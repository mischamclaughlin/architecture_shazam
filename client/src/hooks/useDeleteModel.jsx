// ./client/src/hooks/useDeleteModel.jsx
import { useCallback } from 'react';

export function useDeleteModel() {
    return async (id) => {
        if (!id) throw new Error('No model id provided');
        const res = await fetch(`/api/models/${id}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || 'Failed to delete model');
        }
        return res.json();
    };
}
