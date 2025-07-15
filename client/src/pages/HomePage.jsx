// ./client/src/pages/HomePage.jsx
import React, { useState } from 'react';

import DragDropUpload from '../components/DragDropUpload';
import StatusMessage from '../components/StatusInfo';
import ErrorStatusMessage from '../components/ErrorStatus';
import ImageGallery from '../components/ImageGallery';

import { useGenerateImage } from '../hooks/useGenerateImage'
import { loadMyImages } from '../hooks/loadMyImages';

import './HomePage.css'


export default function HomePage() {
    const [file, setFile] = useState(null)
    const { status, imageUrl, errorStatus, generate } = useGenerateImage();
    const { images, loading, reload } = loadMyImages();

    const handleGenerate = async (file) => {
        await generate(file);
        reload();
    };

    return (
        <>
            <div className="file-upload-area">
                <h2>Upload File</h2>
                <DragDropUpload onFileSelect={setFile} />
                <button
                    onClick={() => handleGenerate(file)}
                    disabled={!file || status.includes('...')}
                    className='generate-button'
                >
                    {status && status != 'Done!' ? 'Generating…' : 'Generate Image'}
                </button>
            </div>

            {errorStatus
                ? <ErrorStatusMessage status={errorStatus} />
                : <StatusMessage status={status} />
            }

            {!loading && images.length > 0 && (
                <div className='gallery-area'>
                    <h3>Your Image Gallery</h3>
                    <ImageGallery images={images} />
                </div>
            )}
        </>
    );
}