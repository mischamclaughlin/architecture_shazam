// ./client/src/components/StatusInfo.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './StatusInfo.css';

const DEFAULT_TIPS = [
    'This can take a few minutes…',
    'Bigger files take longer to process.',
    'Network speed affects this step.',
    'Rendering may spike your CPU/GPU briefly.',
];

export default function StatusMessage({ status, tips = DEFAULT_TIPS, intervalMs = 7500 }) {
    const [show, setShow] = useState(false);
    const [tipIdx, setTipIdx] = useState(0);
    const timerRef = useRef(null);

    const isDone = status === 'Done!';
    const shouldRotate = !!status && !isDone;

    // Reset / show on status change
    useEffect(() => {
        if (!status) {
            setShow(false);
            setTipIdx(0);
            return;
        }
        setShow(true);

        // Auto-hide after a moment when done
        if (isDone) {
            const t = setTimeout(() => setShow(false), 2500);
            return () => clearTimeout(t);
        }
    }, [status, isDone]);

    // Rotate helper tips while working
    useEffect(() => {
        if (!shouldRotate) {
            if (timerRef.current) clearInterval(timerRef.current);
            timerRef.current = null;
            return;
        }
        // start/refresh interval
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
            setTipIdx(i => (i + 1) % Math.max(tips.length, 1));
        }, intervalMs);

        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            timerRef.current = null;
        };
    }, [shouldRotate, tips, intervalMs]);

    const tip = useMemo(() => (shouldRotate && tips.length ? tips[tipIdx] : ''), [shouldRotate, tips, tipIdx]);

    if (!status) return null;

    return (
        <p className={`status-success${show ? ' show' : ''}`}>
            <span className="status-main">{status}</span>
            {tip && <span className="status-sep"> · </span>}
            {tip && <span className="status-tip">{tip}</span>}
        </p>
    );
}