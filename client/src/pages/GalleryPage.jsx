// ./client/src/pages/GalleryPage.jsx
import React from 'react';
import ImageGallery from '../components/ImageGallery';
import ModelGallery from '../components/ModelGallery';
import { loadMyImages } from '../hooks/loadMyImages';
import { loadMyModels } from '../hooks/loadMyModels';
import "./GalleryPage.css";

export default function GalleryPage() {
    const { images, loadingImages } = loadMyImages();
    const { models, loadingModels, reloadModels } = loadMyModels();

    return (
        <div className="gallery-page">
            {!loadingImages && images.length > 0 && (
                <details className="gallery-area" open>
                    <summary>Image Gallery</summary>
                    <div className="gallery-content">
                        <ImageGallery images={images} />
                    </div>
                </details>
            )}

            {!loadingModels && models.length > 0 && (
                <details className="gallery-area" open>
                    <summary>Model Gallery</summary>
                    <div className="gallery-content">
                        <ModelGallery models={models} onReload={reloadModels} />
                    </div>
                </details>
            )}

            {!loadingImages && !images.length && !loadingModels && !models.length && (
                <p>No images or models yet.</p>
            )}
        </div>
    );
}