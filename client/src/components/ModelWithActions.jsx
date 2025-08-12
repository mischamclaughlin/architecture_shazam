// ./client/src/components/ModelWithActions.jsx
import React, { useMemo } from 'react';
import '@google/model-viewer';
import ModelViewer from './ModelViewer';
import { HiOutlineDownload } from 'react-icons/hi';
import { MdOutlineDeleteForever } from 'react-icons/md';
import { useDeleteModel } from '../hooks/useDeleteModel';
import './ModelWithActions.css';

export default function ModelWithActions({
    id,
    url,
    filename,
    onDeleted,
    variant = 'default',
    width,
    height,
}) {
    const deleteModel = useDeleteModel();

    const isGLB = useMemo(() => {
        if (!url && !filename) return false;
        return /\.glb($|\?)/i.test(url || '') || /\.glb$/i.test(filename || '');
    }, [url, filename]);

    const title = (filename || '')
        .split('/').pop()
        .replace(/\.(png|jpe?g|webp|obj|glb|fbx)$/i, '')
        .replace(/_(thumb|preview)$/i, '')
        .replace(/_\d{8}_\d{6}$/, '')
        .replace(/[_\.]+/g, ' ')
        .trim();

    const handleDelete = async () => {
        try {
            await deleteModel(id);
            onDeleted?.();
        } catch (e) {
            console.error('Delete failed:', e);
            alert('Could not delete model: ' + (e.message || e));
        }
    };

    const wrapperStyle = {
        width: width ? `${width}px` : undefined,
        height: height ? `${height}px` : undefined,
    };

    return (
        <div className={`mwa-wrapper ${variant}`} style={wrapperStyle}>
            <div className="mwa-stage">
                {isGLB ? (
                    <model-viewer
                        src={url}
                        alt={title}
                        camera-controls
                        ar
                        shadow-intensity="1"
                        exposure="1"
                        style={{ width: '100%', height: '100%', display: 'block' }}
                    />
                ) : (
                    <ModelViewer url={url} />
                )}
            </div>

            {/* Hide the overlay in compact mode */}
            {variant !== 'compact' && (
                <div className="mwa-overlay">
                    <div className="mwa-title">{title || filename}</div>
                    <div className="mwa-actions">
                        <a href={url} download={filename} className="mwa-button" title="Download">
                            <HiOutlineDownload />
                        </a>
                        <button onClick={handleDelete} className="mwa-button" title="Delete">
                            <MdOutlineDeleteForever />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}