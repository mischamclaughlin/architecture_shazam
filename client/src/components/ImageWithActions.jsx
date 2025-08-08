// ./client/src/components/ImageWithActions.jsx
import React, { useState } from "react";

import { useDeleteImage } from "../hooks/useDeleteImage";

import { HiOutlineDownload } from 'react-icons/hi';
import { BiFullscreen } from 'react-icons/bi';
import { MdOutlineDeleteForever } from "react-icons/md";

import './ImageWithActions.css';

export default function ImageWithActions({ id, src, alt = '', onDeleted }) {
    const [zoomed, setZoomed] = useState(false)
    const deleteImage = useDeleteImage();
    const title = alt.split(/mp3|wav|png|m4a/)[0].replaceAll('_', ' ').replaceAll('.', '').slice(0, -15)

    const handleDelete = async () => {
        try {
            await deleteImage(id);
            onDeleted?.();
        } catch (e) {
            console.error('Delete failed: ', e);
            alert('Could not delete image: ' + e.message);
        }
    };

    return (
        <div className='image-area'>
            <div className="iwa-wrapper">
                <img className="iwa-img" src={src} alt={alt} />
                <div className="iwa-overlay">
                    <div className="iwa-title">
                        <p>{title}</p>
                    </div>
                    <div className="iwa-actions">
                        <a href={src} download={alt} className="iwa-button">
                            <HiOutlineDownload />
                        </a>
                        <button onClick={handleDelete} className="iwa-button">
                            <MdOutlineDeleteForever />
                        </button>
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