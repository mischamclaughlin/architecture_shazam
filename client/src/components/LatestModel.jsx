// ./client/src/components/LatestModel.jsx
import React from 'react';
import { useLatestModel } from '../hooks/useLatestModel';
import ModelWithActions from './ModelWithActions';
import './LatestModel.css';

export default function LatestModel() {
    const { model, loading, reload } = useLatestModel();

    if (loading) return <p>Loading your latest 3D model…</p>;
    if (!model) return <p>No model generated yet.</p>;

    return (
        <div className='latest-model-area'>
            <h2 className='title'>Latest Model</h2>
            <div className='model'>
                <ModelWithActions
                    id={model.id}
                    url={model.url}
                    filename={model.filename}
                    onDeleted={reload}
                />
            </div>
        </div>
    );
}