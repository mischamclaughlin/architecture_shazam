// ./client/src/pages/HomePage.jsx
import React, { useState } from 'react';
import DragDropUpload from '../components/DragDropUpload';
import SpotifySearch from '../components/SongSearch';
import StatusMessage from '../components/StatusInfo';
import ErrorStatusMessage from '../components/ErrorStatus';
import LatestImage from '../components/LatestImage';
import LatestModel from '../components/LatestModel';

import { useGenerate } from '../hooks/useGenerate';

import './HomePage.css';

export default function HomePage() {
    const [file, setFile] = useState(null);
    const [snippetInfo, setSnippetInfo] = useState(null);
    const [query, setQuery] = useState('');
    const [refreshKey, setRefreshKey] = useState(0);
    const [option, setOption] = useState('house')

    const { status, errorStatus, generate, generateFromUrl } = useGenerate();

    const handleTrackLookup = ({ title, artist, preview_url }) => {
        setQuery(title);
        setSnippetInfo({ title, artist, preview_url });
        setFile(null);
    };

    const handleGenerateClick = async (type, option) => {
        try {
            if (snippetInfo?.preview_url) {
                await generateFromUrl(type, option, snippetInfo);
            } else if (file) {
                await generate(type, option, { file, title: query });
            }
        } catch (e) {
            console.error('Generation failed: ', e);
        };

        setFile(null);
        setSnippetInfo(null);
        setQuery('');
        setRefreshKey(k => k + 1);
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
                    <h2>Song Upload</h2>
                    <DragDropUpload
                        onFileSelect={setFile}
                    />
                </div>
                <div className='select-area'>
                    <label className='title-selector'>Building Type:</label>
                    <select className='option-selector' value={option} onChange={e => setOption(e.target.value)}>
                        <option value="house">house</option>
                        <option value="skyscraper">skyscraper</option>
                        <option value="apartments">apartments</option>
                    </select>
                </div>
                <div className='btn-area'>
                    <button
                        onClick={() => handleGenerateClick('image', option)}
                        disabled={
                            (!file && !(snippetInfo?.preview_url)) ||
                            status.includes('…')
                        }
                        className="generate-button"
                    >
                        {status && status !== 'Done!' ? 'Generating…' : 'Generate Image'}
                    </button>
                    <button
                        onClick={() => handleGenerateClick('model', option)}
                        disabled={
                            (!file && !(snippetInfo?.preview_url)) ||
                            status.includes('…')
                        }
                        className="generate-button"
                    >
                        {status && status !== 'Done!' ? 'Generating…' : 'Generate Model'}
                    </button>
                </div>

                {errorStatus
                    ? <ErrorStatusMessage status={errorStatus} />
                    : <StatusMessage status={status} />
                }
            </div>
            <div className='latest-area'>
                <details className='view-generation' open>
                    <summary>Latest Image</summary>
                    <div>
                        <LatestImage key={refreshKey} />
                    </div>
                </details>
                <details className='view-generation' open>
                    <summary>Latest Model</summary>
                    <div>
                        <LatestModel key={refreshKey} />
                    </div>
                </details>
            </div>
        </div >
    );
}