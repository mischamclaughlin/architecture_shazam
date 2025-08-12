// client/src/hooks/useGenerateImage.jsx
import { useState, useCallback } from 'react';

export function useGenerate() {
  const [status, setStatus] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [errorStatus, setErrorStatus] = useState('');

  const generate = useCallback(async (type, option, params) => {
    const { file, title, artist, album, release } = params;

    setErrorStatus('');
    setImageUrl('');

    if (!file) {
      setErrorStatus('Please select a file first');
      return;
    }

    // Build one FormData for analyse
    const formData = new FormData();
    formData.append('file', file);
    formData.append('action', type);
    formData.append('building', option);
    if (title) formData.append('title', title);
    if (artist) formData.append('artist', artist);
    if (album) formData.append('album', album);
    if (release) formData.append('release', release);

    try {
      // 1) Analyse
      setStatus('Extracting features…');
      let res = await fetch('/api/analyse', { method: 'POST', body: formData });
      let json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Analyse failed');

      const analysisId = json.analysisId;

      // 2) Describe
      setStatus('Generating description…');
      res = await fetch('/api/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId,
          building_type: option
        }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Describe failed');

      const promptId = json.promptId;

      // 3) Render
      setStatus('Rendering image…');
      res = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          promptId,
          action: type
        }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Render failed');

      // Success!
      setImageUrl(json.imageUrl);
      setStatus('Done!');
    } catch (err) {
      console.error('generate error:', err);
      setStatus('');
      setErrorStatus(err.message || 'An unexpected error occurred');
    }
  }, []);

  /*
   * Kick off the server pipeline by downloading a snippet URL,
   * turning it into a File, plus passing along all its metadata.
   */
  const generateFromUrl = useCallback(async (type, option, snippetInfo) => {
    setErrorStatus('');
    setImageUrl('');

    try {
      // Download the 30s clip
      setStatus('Downloading preview…');
      const clipRes = await fetch(snippetInfo.preview_url);
      if (!clipRes.ok) throw new Error('Failed to download snippet');
      const blob = await clipRes.blob();
      const ext = snippetInfo.preview_url
        .split('.')
        .pop()
        .split('?')[0];
      const file = new File([blob], `snippet.${ext}`, { type: blob.type });

      // Build the same FormData with all metadata fields
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', snippetInfo.title);
      formData.append('artist', snippetInfo.artist);
      formData.append('action', type);
      formData.append('building', option);
      if (snippetInfo.album) formData.append('album', snippetInfo.album);
      if (snippetInfo.release) formData.append('release', snippetInfo.release);

      // 1) Analyse
      setStatus('Extracting features…');
      let res = await fetch('/api/analyse', { method: 'POST', body: formData });
      let json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Analyse failed');
      const analysisId = json.analysisId;

      // 2) Describe
      setStatus('Generating description…');
      res = await fetch('/api/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId,
          building_type: option
        }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Describe failed');
      const promptId = json.promptId;

      // 3) Render
      setStatus(`Rendering ${type}...`);
      res = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          promptId,
          action: type
        }),
      });
      json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Render failed');

      setImageUrl(json.imageUrl);
      setStatus('Done!');
    } catch (err) {
      console.error('generateFromUrl error:', err);
      setStatus('');
      setErrorStatus(err.message || 'Could not generate image');
    }
  }, []);

  return { status, imageUrl, errorStatus, generate, generateFromUrl };
}