// ./client/src/components/ImageWithActions.jsx
import React, { useState } from "react";

import { HiOutlineDownload } from 'react-icons/hi';
import { BiFullscreen } from 'react-icons/bi';

import './ImageWithActions.css';

export default function ImageWithActions({ src, alt = '' }) {
    const [zoomed, setZoomed] = useState(false)
    const title = alt.split(/mp3|wav|png|m4a/)[0].replaceAll('_', ' ').replaceAll('.', '')

    return (
        <div className='image-area'>
            <div className="iwa-wrapper">
                <img className="iwa-img" src={src} alt={alt} />
                <div className="iwa-overlay">
                    <div className="iwa-title">
                        <p>{title}</p>
                    </div>
                    <div className="iwa-actions">
                        <a href={src} download className="iwa-button">
                            <HiOutlineDownload />
                        </a>
                        <button
                            type="button"
                            onClick={() => setZoomed(true)}
                            className="iwa-button"
                        >
                            <BiFullscreen />
                        </button>
                    </div>
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