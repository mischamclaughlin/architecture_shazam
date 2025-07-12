// ./client/src/pages/HomePage.jsx
import React, { useState } from 'react';

import DragDropUpload from '../components/DragDropUpload';
import StatusMessage from '../components/StatusInfo';
import ImageWithActions from '../components/ImageWithActions';

import { useGenerateImage } from '../hooks/useGenerateImage'

import './HomePage.css'


export default function HomePage() {
    const [file, setFile] = useState(null)
    const { status, imageUrl, generate } = useGenerateImage();

    return (
        <div>
            <div className="file-upload-area">
                <h2>Upload File</h2>
                <DragDropUpload onFileSelect={setFile} />
                <button
                    onClick={() => generate(file)}
                    disabled={!file || status.includes('...')}
                    className='generate-button'
                >
                    {status && status != 'Done!' ? 'Generating…' : 'Generate Image'}
                </button>
            </div>
            <StatusMessage status={status} />
            {imageUrl && (
                <ImageWithActions src={imageUrl} alt='generated result' />
            )}
        </div>
    );
}
