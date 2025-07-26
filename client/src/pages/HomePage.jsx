import React, { useState } from 'react';
import DragDropUpload from '../components/DragDropUpload';
import SpotifySearch from '../components/SongSearch';
import StatusMessage from '../components/StatusInfo';
import ErrorStatusMessage from '../components/ErrorStatus';
import ImageGallery from '../components/ImageGallery';
import { useGenerateImage } from '../hooks/useGenerateImage';
import { loadMyImages } from '../hooks/loadMyImages';
import './HomePage.css';

export default function HomePage() {
    const [file, setFile] = useState(null);
    const [snippetInfo, setSnippetInfo] = useState(null);
    const [query, setQuery] = useState('');

    const { status, errorStatus, generate, generateFromUrl } = useGenerateImage();
    const { images, loading, reload } = loadMyImages();

    const handleTrackLookup = ({ title, artist, preview_url }) => {
        setQuery(title);
        setSnippetInfo({ title, artist, preview_url });
        setFile(null);
    };

    const handleGenerateClick = async () => {
        if (snippetInfo?.preview_url) {
            await generateFromUrl(snippetInfo);
        } else if (file) {
            await generate({ file, title: query });
        }

        setFile(null);
        setSnippetInfo(null);
        setQuery('');
        reload();
    };

    const uploadKey = file
        ? file.name
        : snippetInfo?.preview_url
            ? snippetInfo.title
            : 'empty';

    return (
        <div className="home-page">
            <div className="file-interaction-area">
                <div className="song-search-area">
                    <h2>Song Search</h2>
                    <SpotifySearch
                        value={query}
                        onChange={setQuery}
                        onResult={handleTrackLookup}
                    />
                </div>

                <div className="file-upload-area">
                    <h2>Upload File</h2>
                    <DragDropUpload
                        key={uploadKey}
                        onFileSelect={setFile}
                    />
                </div>

                <button
                    onClick={handleGenerateClick}
                    disabled={
                        (!file && !(snippetInfo?.preview_url)) ||
                        status.includes('…')
                    }
                    className="generate-button"
                >
                    {status && status !== 'Done!' ? 'Generating…' : 'Generate Image'}
                </button>

                {errorStatus
                    ? <ErrorStatusMessage status={errorStatus} />
                    : <StatusMessage status={status} />
                }
            </div>

            {!loading && images.length > 0 && (
                <div className="gallery-area">
                    <h3>Your Image Gallery</h3>
                    <ImageGallery images={images} />
                </div>
            )}
        </div>
    );
}