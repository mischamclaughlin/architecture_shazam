// client/src/hooks/useGenerateImage.jsx
import { useState, useCallback } from 'react';

export function useGenerateImage() {
  const [status, setStatus] = useState('');
  const [imageUrl, setImageUrl] = useState('');

  const generate = useCallback(async file => {
    if (!file) {
      setStatus('Please select a file first');
      return;
    }

    // 1) Analyze
    setStatus('Extracting features…');
    const form = new FormData();
    form.append('file', file);
    let res = await fetch('/api/analyse', { method: 'POST', body: form });
    let json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Analyse failed');
    const analysisId = json.analysisId;

    // 2) Describe
    setStatus('Generating description…');
    res = await fetch('/api/describe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysisId }),
    });
    json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Describe failed');
    const promptId = json.promptId;

    // 3) Render
    setStatus('Rendering image…');
    res = await fetch('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ promptId }),
    });
    json = await res.json();
    if (!res.ok) throw new Error(json.error || 'Render failed');

    // Success
    setImageUrl(json.imageUrl);
    setStatus('Done!');
  }, []);

  return { status, imageUrl, generate };
}