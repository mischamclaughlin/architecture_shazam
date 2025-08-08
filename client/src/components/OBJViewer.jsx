// ./client/src/components/OBJViewer.jsx
import React, { Suspense } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { OrbitControls, Bounds, Html } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import './OBJViewer.css';

function StaticModel({ url }) {
    const obj = useLoader(OBJLoader, url);
    return (
        <group>
            <axesHelper args={[1]} />
            <primitive object={obj} />
        </group>
    );
}

export default function OBJViewer({ url }) {
    return (
        <div className="model-canvas">
            <Canvas
                camera={{ position: [0, -90, 6], fov: 75 }}>
                <ambientLight intensity={0.8} />
                <directionalLight position={[10, 10, 10]} intensity={1} />
                <directionalLight position={[-10, -10, 5]} intensity={0.5} />

                <Suspense fallback={<Html center>Loading 3D…</Html>}>
                    <Bounds fit clip margin={1}>
                        <StaticModel url={url} />
                    </Bounds>
                </Suspense>

                <OrbitControls
                    enableDamping
                    dampingFactor={0.1}
                    rotateSpeed={0.7}
                    zoomSpeed={0.6}
                    panSpeed={1.2}
                    screenSpacePanning
                    minDistance={1}
                    maxDistance={12}
                    target={[0, 0, 0]}
                />
            </Canvas>
        </div>
    );
}