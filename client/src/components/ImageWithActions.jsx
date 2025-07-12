// ./client/src/components/ImageWithActions.jsx
import React, { useState } from "react";
import './ImageWithActions.css'

export default function ImageWithActions({ src, alt = '' }) {
    const [zoomed, setZoomed] = useState(false)

    return (
        <div className='image-area'>
            <h3>Your generated Image</h3>
            <div className="iwa-wrapper">
                <img className="iwa-img" src={src} alt={alt} />
                <div className="iwa-overlay">
                    <a href={src} download className="iwa-button">Download</a>
                    <button
                        type="button"
                        onClick={() => setZoomed(true)}
                        className="iwa-button"
                    >
                        Zoom
                    </button>
                </div>
            </div>

            {zoomed && (
                <div className='iwa-modal' onClick={() => setZoomed(false)}>
                    <img className="iwa-modal-img" src={src} alt={alt} />
                </div>
            )}
        </div>
    );
}