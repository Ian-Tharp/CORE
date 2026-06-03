// planet-materials.ts — material factories. All colours converted to LINEAR.
import * as THREE from 'three';
import type { WorldGenParams } from './world-gen-params';
import { atmosphereVert, atmosphereFrag, cloudVert, cloudFrag } from './atmosphere.glsl';

/** Surface uses vertex colours so it composes with the scene's lights/shadows. */
export function makeSurfaceMaterial(_p: WorldGenParams): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.92,
    metalness: 0.0,
    flatShading: false
  });
}

export function makeWaterMaterial(p: WorldGenParams): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setStyle(p.palette.water).convertSRGBToLinear(),
    transparent: true,
    opacity: 0.8,
    roughness: 0.16,
    metalness: 0.15,
    depthWrite: false
  });
  // Kill shoreline z-fighting against the surface.
  mat.polygonOffset = true;
  mat.polygonOffsetFactor = -1;
  mat.polygonOffsetUnits = -1;
  return mat;
}

export function makeAtmosphereMaterial(p: WorldGenParams): THREE.ShaderMaterial {
  const inner = new THREE.Color(p.biomeArchetype === 'astral' ? '#b98cff' : '#7fe3ff').convertSRGBToLinear();
  const outer = new THREE.Color('#1b3a6b').convertSRGBToLinear();
  return new THREE.ShaderMaterial({
    vertexShader: atmosphereVert,
    fragmentShader: atmosphereFrag,
    transparent: true,
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColorInner: { value: inner },
      uColorOuter: { value: outer },
      uSunDir: { value: new THREE.Vector3(0.6, 0.5, 0.4).normalize() },
      uIntensity: { value: 1.3 },
      uPower: { value: 4.0 }
    }
  });
}

export function makeCloudMaterial(_p: WorldGenParams): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: cloudVert,
    fragmentShader: cloudFrag,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: new THREE.Color('#eef6ff').convertSRGBToLinear() },
      uIntensity: { value: 0.85 }
    }
  });
}
