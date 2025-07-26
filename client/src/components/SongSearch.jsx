// ./client/src/components/SpotifySearch.jsx
import React, { useState } from "react";

import ErrorStatusMessage from "./ErrorStatus";

import './SongSearch.css'

export default function SpotifySearch({ onResult }) {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [info, setInfo] = useState(null);

    const handleSubmit = async e => {
        e.preventDefault();
        if (!query.trim()) {
            setError('Please enter a song name');
            return;
        }
        setError('');
        setInfo(null);
        setLoading(true);

        try {
            const res = await fetch('/api/track_snippet', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ songName: query }),
            });
            const json = await res.json();
            if (!res.ok) throw new Error(json.error || 'Lookup Failed');
            setInfo(json);
            onResult?.(json);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="spotify-search">
            <form className="search-area" onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Search song"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Searching...' : 'Lookup'}
                </button>
            </form>


            {info && (
                <div className="song-info">
                    <h4>{info.title} - {info.artist}</h4>
                    {info.preview_url
                        ? <audio controls src={info.preview_url} />
                        : <p>No preview available</p>
                    }
                </div>
            )}

            {error && <ErrorStatusMessage status={error} />}
        </div>
    )
}
