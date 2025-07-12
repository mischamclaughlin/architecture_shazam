// src/components/DragDropUpload.jsx
import React, { useRef, useState } from 'react';
import './DragDropUpload.css';

export default function DragDropUpload({
    onFileSelect,
    accept = '.mp3,.wav'
}) {
    const inputRef = useRef();
    const [fileName, setFileName] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [error, setError] = useState('');

    const isValidAudio = file => {
        const allowedTypes = ['audio/mpeg', 'audio/wav'];
        const allowedExts = ['.mp3', '.wav'];
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
        onFileSelect(file);
    };

    const handleClick = () => inputRef.current.click();
    const handleChange = e => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    };

    const handleDrag = e => {
        e.preventDefault(); e.stopPropagation();
        setDragActive(e.type === 'dragenter' || e.type === 'dragover');
    };
    const handleDrop = e => {
        e.preventDefault(); e.stopPropagation();
        setDragActive(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
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
                    accept={accept}
                    style={{ display: 'none' }}
                    onChange={handleChange}
                />
                {fileName
                    ? <p className="file-name">Selected: <strong>{fileName}</strong></p>
                    : <p>Click or drag an MP3/WAV here to select</p>
                }
            </div>
            {error && <p className="error">{error}</p>}
        </div>
    );
}