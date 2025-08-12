// ./client/src/components/ModelGallery.jsx
import React from "react";
import { loadMyModels } from "../hooks/loadMyModels";
import ModelWithActions from "./ModelWithActions";
import './ImageGallery.css';

export default function ModelGallery() {
    const { models, loadingModels, reloadModels } = loadMyModels();

    if (loadingModels) return <p>Loading your gallery...</p>;
    if (!models.length) return <p>No Models generated yet.</p>;

    return (
        <div className="gallery-grid">
            {models.map(m => (
                <div key={m.id} className="gallery-item">
                    <ModelWithActions
                        id={m.id}
                        url={m.url}
                        filename={m.filename}
                        onDeleted={reloadModels}
                    />
                </div>
            ))}
        </div>
    );
}