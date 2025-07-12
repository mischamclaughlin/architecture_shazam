// ./client/src/components/StatusInfo.jsx
import React, { useEffect, useState } from 'react';
import './StatusInfo.css';

export default function StatusMessage({ status }) {
    const [show, setShow] = useState(false);

    useEffect(() => {
        if (!status) {
            setShow(false);
            return;
        }
        setShow(true);
        if (status === 'Done!') {
            const t = setTimeout(() => setShow(false), 2500);
            return () => clearTimeout(t);
        }
    }, [status]);

    return (
        <p className={`status${show ? ' show' : ''}`}>
            {status}
        </p>
    );
}