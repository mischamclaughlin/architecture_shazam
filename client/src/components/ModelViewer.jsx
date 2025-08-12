// ./client/src/components/ModelViewer.jsx
import React, { Suspense } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { OrbitControls, Bounds, Html, Environment } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
// import './ModelViewer.css';

function AnyModel({ url }) {
    const ext = (url.split('.').pop() || '').toLowerCase().split('?')[0];

    if (ext === 'obj') {
        const obj = useLoader(OBJLoader, url);
        return (
            <group>
                <primitive object={obj} />
            </group>
        );
    }

    // default: glb/gltf
    const gltf = useLoader(GLTFLoader, url);
    return (
        <group>
            <primitive object={gltf.scene} />
        </group>
    );
}

export default function ModelViewer({ url, height = 420 }) {
    return (
        <div className="model-canvas" style={{ height }}>
            <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
                {/* Lighting */}
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 10]} intensity={1} />
                <Environment preset="city" />

                {/* Lazy-load model */}
                <Suspense fallback={<Html center>Loading 3D…</Html>}>
                    {/* Bounds auto-centers and frames the model */}
                    <Bounds fit clip margin={0.3} observe>
                        <AnyModel url={url} />
                    </Bounds>
                </Suspense>

                <OrbitControls
                    enableDamping
                    dampingFactor={0.1}
                    rotateSpeed={0.7}
                    zoomSpeed={0.6}
                    panSpeed={1.2}
                    screenSpacePanning
                    minDistance={2}
                    maxDistance={12}
                    target={[0, 0, 0]}
                />
            </Canvas>
        </div>
    );
}