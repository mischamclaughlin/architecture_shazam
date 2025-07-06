// src/components/DragDropUpload.jsx
import React, { useRef, useState } from 'react';
import './DragDropUpload.css'

export default function DragDropUpload({ uploadUrl = '/api/upload' }) {
    const inputRef = useRef();
    const [fileName, setFileName] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [error, setError] = useState('');

    const isValidAudio = file => {
        const allowedTypes = ['audio/mpeg', 'audio/wav'];
        const allowedExts = ['.mp3', '.wav']
        const { type, name } = file;
        const ext = name.slice(name.lastIndexOf('.')).toLowerCase();
        return allowedTypes.includes(type) && allowedExts.includes(ext);
    };

    const handleFile = file => {
        if (!isValidAudio(file)) {
            setError('Please upload an MP3 or WAV file');
            return;
        }
        setError('');
        setFileName(file.name);
        uploadFile(file);
    };

    const handleClick = () => inputRef.current.click();
    const handleChange = e => {
        const file = e.target.files[0];
        if (file) handleFile(file)
    };

    const handleDrag = e => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = e => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file)
    };

    const uploadFile = async file => {
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch(uploadUrl, {
                method: 'POST',
                body: form,
            });
            if (!res.ok) throw new Error(`Status ${res.status}`);
        } catch (err) {
            console.error(err);
            alert('Upload failed');
        }
    };

    return (
        <div>
            <div
                className={dragActive ? 'dropzone active' : 'dropzone'}
                onClick={handleClick}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept='.mp3, audio/mpeg, .wav, audio/wav'
                    style={{ display: 'none' }}
                    onChange={handleChange}
                />
                {fileName
                    ? <p className='file-name'>Selected file: <strong>{fileName}</strong></p>
                    : <p>Click or drag an MP3/WAV file here to upload</p>
                }
            </div>
            {error && (
                <p className='error'>
                    {error}
                </p>
            )}
        </div>
    );
}