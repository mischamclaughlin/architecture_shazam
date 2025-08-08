// ./client/src/components/LatestImage.jsx
import React, { useEffect } from 'react';
import { useLatestImage } from '../hooks/useLatestImage';
import ImageWithActions from './ImageWithActions';
import './LatestImage.css';

export default function LatestImage({ refreshKey }) {
    const { image: img, loading, reload } = useLatestImage();

    useEffect(() => {
        reload();
    }, [refreshKey, reload]);

    if (loading) return <p>Loading your latest image…</p>;
    if (!img) return <p>No images generated yet.</p>;

    return (
        <div className='latest-image-area'>
            <h2 className='title'>Latest Image</h2>
            <div className='image'>
                <ImageWithActions
                    key={img.id}
                    id={img.id}
                    src={img.url}
                    alt={img.filename}
                    onDeleted={() => reload()}
                />
            </div>
        </div>
    );
}