// ./client/src/components/ImageGallery.jsx
import React from "react";
import { loadMyImages } from "../hooks/loadMyImages";
import ImageWithActions from "./ImageWithActions";
import './ImageGallery.css'


export default function ImageGallery() {
    const { images, loadingImages, reloadImages } = loadMyImages();

    if (loadingImages) return <p>Loading your gallery...</p>;
    if (images.length === 0) return <p>No images generated yet.</p>

    return (
        <div className="gallery-grid">
            {images.map(img => (
                <div key={img.id} className="gallery-item">
                    <ImageWithActions
                        key={img.id}
                        id={img.id}
                        src={img.url}
                        alt={img.filename}
                        onDeleted={() => reloadImages()}
                    />
                </div>
            ))}
        </div>
    );
}