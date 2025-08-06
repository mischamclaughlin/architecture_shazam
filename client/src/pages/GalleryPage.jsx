// ./client/src/pages/ImageGalleryPage.jsx
import React from 'react';
import ImageGallery from '../components/ImageGallery';
import { loadMyImages } from '../hooks/loadMyImages';

import "./GalleryPage.css"

export default function GalleryPage() {
    const { images, loading, reload } = loadMyImages();

    return (
        <div className='gallery-page'>
            {!loading && images.length > 0 && (
                <details className="gallery-area">
                    <summary>Image Gallery</summary>
                    <ImageGallery images={images} />
                </details>
            )}
        </div>
    )
}
