import React, { createElement } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

import wireLogo from './assets/sugar_icon_0.svg'
import textureLogo from './assets/sugar_icon_1bis.svg'
import hybridLogo from './assets/sugar_icon_2.svg'

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

import {DropInViewer} from '@mkkellogg/gaussian-splats-3d';

import data from './scene_to_load.json';

// SuGaR outputs
console.log(data);
const ply_path = '../' + data['ply_path'];
const obj_path = '../' + data['obj_path'];
const png_path = '../' + data['png_path'];

// Set up the HUD
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(
    <React.StrictMode>
        <div className="banner">
        </div>
        <div className="hudstill" id="button0">
            SuGaR - Viewer
        </div>
        <div className="hud" id="button">
            Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering
        </div>
        
        <div className="logo hybrid">
            <img src={hybridLogo} className="logoimg" alt="Hybrid representation logo" />
            <br />Hybrid
        </div>

        <div className="logo texture">
            <img src={textureLogo} className="logoimg" alt="Textured mesh logo" />
            <br />Textured
        </div>

        <div className="logo wire">
            <img src={wireLogo} className="logoimg" alt="Wireframed mesh logo" />
            <br />Wireframe
        </div>

        <div className="footer"> 
            Built by <a href="https://anttwo.github.io/">Antoine Guédon</a> with <a href="https://vitejs.dev/">Vite</a> and <a href="https://threejs.org/">Three.js</a>; 
            Huge thanks to <a href="https://github.com/mkkellogg/GaussianSplats3D">Mark Kellogg</a> for his 3D Gaussian Splatting implementation for Javascript.
        </div>
    </React.StrictMode>,);

// Scene and camera
const threeScene = new THREE.Scene();
threeScene.background = new THREE.Color( '#1A1A1A' );
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const ambientLight = new THREE.AmbientLight( 0xffffff, 3.0 );
// const ambientLight = new THREE.AmbientLight();
threeScene.add( ambientLight );

// Renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
// renderer.outputEncoding = THREE.sRGBEncoding;
document.body.appendChild( renderer.domElement );
// renderer.toneMappingExposure = 0.5;
// THREE.ColorManagement.legacyMode = false

// Controls
const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true

// Load the splat scene
var viewer = new DropInViewer({
    'gpuAcceleratedSort': true,
    'sharedMemoryForWorkers': false,
});
viewer.addSplatScene(
    // "./media/hybrid/qant03/qant03_7k.ply", 
    // "./src/media/hybrid/qant03/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply",
    // "./../output/refined_ply/dukemon3/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.ply",
    ply_path,
{
    'splatAlphaRemovalThreshold': 5,
    'showLoadingSpinner': true,
});
viewer.rotation.x = Math.PI;
threeScene.add(viewer);

// Load the mesh and UV texture
const objLoader = new OBJLoader();
const textureLoader = new THREE.TextureLoader();
const [ texture, obj ] = await Promise.all( [
    // textureLoader.loadAsync( './src/media/obj/qant03/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.png' ),
    // objLoader.loadAsync( './src/media/obj/qant03/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj' ),
    // textureLoader.loadAsync( "./../output/refined_mesh/dukemon3/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.png" ),
    // objLoader.loadAsync( "./../output/refined_mesh/dukemon3/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.obj" ),
    textureLoader.loadAsync( png_path ),
    objLoader.loadAsync( obj_path ),
] );
// texture.encoding = THREE.sRGBEncoding;
texture.colorSpace = THREE.SRGBColorSpace;
obj.traverse( function ( child ) {

    if ( child.isMesh ) {
         child.material.map = texture;
         child.material.toneMapped = false
         child.geometry.computeVertexNormals();
         child.material.side = THREE.DoubleSide;

     }

} );
texture.minFilter = THREE.NearestFilter;
texture.magFilter = THREE.NearestFilter;
obj.rotation.x = Math.PI;
threeScene.add( obj );


// Toggle the visibility of the mesh and the splat scene
viewer.visible = true;
obj.visible = false;
var wireframe = false;

function updateText(mainText, buttonText) {
    const main = document.querySelector('.hudstill');
    main.textContent = mainText;

    const button = document.querySelector('.hud');
    button.textContent = buttonText;
}

function onHybridClick() {
    if (!viewer.visible) {
        viewer.visible = true;
        obj.visible = false;
        var mainText = 'SuGaR: Hybrid representation';
        var buttonText = 'Mesh covered with thin 3D Gaussians, and rendered using Gaussian splatting';
        updateText(mainText, buttonText);
    }
}

function onTextureClick() {
    if (!obj.visible | (obj.visible & wireframe)) {
        wireframe = false;
        obj.traverse( function ( child ) {
            if ( child.isMesh ) {
                child.material.wireframe = false;
            }
        } );
        viewer.visible = false;
        obj.visible = true;
        var mainText = 'SuGaR: Textured mesh';
        var buttonText = 'Mesh covered with a traditional UV texture, and rendered with ambient light';
        updateText(mainText, buttonText);
    }
}

function onWireframeClick() {
    if (!obj.visible | (obj.visible & !wireframe)) {
        wireframe = true;
        obj.traverse( function ( child ) {
            if ( child.isMesh ) {
                child.material.wireframe = true;
            }
        } );
        viewer.visible = false;
        obj.visible = true;
        var mainText = 'SuGaR: Wireframe mesh';
        var buttonText = 'Mesh rendered with wireframe';
        updateText(mainText, buttonText);
    }
}

const hybridButton = document.querySelector('.logo.hybrid');
hybridButton.addEventListener('click', onHybridClick);

const textureButton = document.querySelector('.logo.texture');
textureButton.addEventListener('click', onTextureClick);

const wireframeButton = document.querySelector('.logo.wire');
wireframeButton.addEventListener('click', onWireframeClick);

// Render
camera.position.z = 5;

function animate() {
    requestAnimationFrame(animate)

    controls.update()

    renderer.render( threeScene, camera );
}

animate();
