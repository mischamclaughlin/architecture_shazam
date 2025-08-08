// ./client/src/components/LatestModel.jsx
import React from 'react';
import { useLatestModel } from '../hooks/useLatestModel';
import OBJViewer from './OBJViewer';
import './LatestModel.css';


export default function LatestModel() {
    const { model, loading } = useLatestModel();

    if (loading) return <p>Loading your latest 3D model…</p>;
    if (!model) return <p>No model generated yet.</p>;

    return (
        <div className='latest-model-area'>
            <h2 className='title'>Latest Model</h2>
            <div className='model'>
                <OBJViewer url={model.url} />
            </div>
        </div>
    );
}
