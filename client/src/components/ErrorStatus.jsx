// ./client/src/components/ErrorStatus.jsx
import React, { useEffect, useState } from 'react';
import { IoWarningOutline, IoCloseCircleOutline } from "react-icons/io5";
import './ErrorStatus.css';

export default function ErrorStatusMessage({ status }) {
    const [rendered, setRendered] = useState(false);
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        if (status) {
            setRendered(true);
            requestAnimationFrame(() => {
                setVisible(true);
            });
            if (status === 'Done!') {
                setTimeout(handleClose, 2500);
            }
        } else {
            handleClose();
        }
    }, [status]);

    function handleClose() {
        setVisible(false);
        setTimeout(() => setRendered(false), 500);
    }

    if (!rendered) return null;

    return (
        <div className={`status ${visible ? 'show' : ''}`}>
            <div className="warning">
                <IoWarningOutline />
            </div>
            <p>{status}</p>
            <button className="close-btn" onClick={handleClose}>
                <IoCloseCircleOutline />
            </button>
        </div>
    );
}