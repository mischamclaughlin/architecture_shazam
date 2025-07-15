// client/src/hooks/useGenerateImage.jsx
import { useState, useCallback } from 'react';

export function useGenerateImage() {
  const [status, setStatus] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [errorStatus, setErrorStatus] = useState('');

  const generate = useCallback(async file => {
    setErrorStatus('')
    setImageUrl('')

    if (!file) {
      setStatus('Please select a file first');
      return;
    }

    try {
      // 1) Analyze
      setStatus('Extracting features…');
      const form = new FormData();
      form.append('file', file);
      let res = await fetch('/api/analyse', { method: 'POST', body: form });
      let json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Analyse failed, please try again or change file');
      const analysisId = json.analysisId;

      // 2) Describe
      setStatus('Generating description…');
      res = await fetch('/api/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysisId }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Describe failed, please try again or change file');
      const promptId = json.promptId;

      // 3) Render
      setStatus('Rendering image…');
      res = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ promptId }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Render failed, please try again or change file');

      // Success
      setImageUrl(json.imageUrl);
      setStatus('Done!');
    } catch (err) {
      console.error('useGenerateImage error:', err);
      setStatus('');
      setErrorStatus(err.message || 'An unexpected error occurred, please try again or change file');
    }
  }, []);

  return { status, imageUrl, errorStatus, generate };
}