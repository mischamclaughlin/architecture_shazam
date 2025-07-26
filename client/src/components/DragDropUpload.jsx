// src/components/DragDropUpload.jsx
import React, {
    forwardRef,
    useRef,
    useState,
    useImperativeHandle
} from 'react';
import './DragDropUpload.css';

const DragDropUpload = forwardRef(function DragDropUpload({
    onFileSelect,
    accept = '.mp3,.wav'
}, ref) {
    const inputRef = useRef();
    const [fileName, setFileName] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [error, setError] = useState('');

    // expose clear() to parent
    useImperativeHandle(ref, () => ({
        clear: () => {
            if (inputRef.current) inputRef.current.value = '';
            setFileName('');
            setError('');
            onFileSelect(null);
        }
    }));

    const isValidAudio = file => {
        const allowedTypes = ['audio/mpeg', 'audio/wav'];
        const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
        return allowedTypes.includes(file.type) && ['.mp3', '.wav'].includes(ext);
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
    const handleChange = e => e.target.files[0] && handleFile(e.target.files[0]);
    const handleDrag = e => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(e.type === 'dragenter' || e.type === 'dragover');
    };
    const handleDrop = e => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]);
    };

    return (
        <div className="upload-area">
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
});

export default DragDropUpload;