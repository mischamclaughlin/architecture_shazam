import React from 'react'
import Header from './components/Header';
import DragDropUpload from './components/DragDropUpload';

function App() {
  return (
    <div>
      <Header />
      <div className='file-upload'>
        <div>
          <h1>Upload File</h1>
          <DragDropUpload uploadUrl='/api/upload' />
        </div>
      </div>
    </div>
  );
}

export default App