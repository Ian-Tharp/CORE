import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';

@Injectable({ providedIn: 'root' })
export class EngineService {
  private renderer?: THREE.WebGLRenderer;
  private scene?: THREE.Scene;
  private camera?: THREE.OrthographicCamera;
  private controls?: MapControls;
  private animationHandle = 0;
  private canvas?: HTMLCanvasElement;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private resizeObserver?: ResizeObserver;
  private isActive = false;

  private interactionTargets: THREE.Object3D[] = [];
  private pressed = new Set<string>();
  private _onPointerDown?: (intersections: THREE.Intersection[], button: number) => void;
  private _onPointerUp?: (intersections: THREE.Intersection[], button: number) => void;

  initialize(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    // Make canvas focusable so keyboard controls require focus/activation
    this.canvas.setAttribute('tabindex', '0');
    this.scene = new THREE.Scene();
    // Let CSS behind the canvas show through (gradient in stage)
    this.scene.background = null;

    const { clientWidth: width, clientHeight: height } = canvas;
    const aspect = width / height;
    const viewHeight = 60; // world units visible vertically
    const halfH = viewHeight / 2;
    const halfW = halfH * aspect;
    this.camera = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.1, 2000);
    this.camera.position.set(0, 100, 0);
    this.camera.lookAt(new THREE.Vector3(0, 0, 0));

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    // Transparent clear; CSS gradient supplies background
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(width || 1, height || 1, false);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.controls = new MapControls(this.camera, canvas);
    this.controls.enableRotate = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.zoomSpeed = 1.0;
    this.controls.minZoom = 0.3;
    this.controls.maxZoom = 10.0;
    // MapControls button mapping: middle pan, wheel zoom; free left/right for interactions
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE; // no-op because rotate disabled
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE; // free for context handling
    this.controls.target.set(0, 0, 0);

    const hemi = new THREE.HemisphereLight(0xbfdfff, 0x1b2727, 1.2);
    this.scene.add(hemi);
    const ambient = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xe8fff7, 1.6);
    dir.position.set(60, 180, 80);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.near = 0.5;
    dir.shadow.camera.far = 500;
    this.scene.add(dir);

    window.addEventListener('resize', this.handleResize, { passive: true });
    canvas.addEventListener('pointermove', this.handlePointerMove, { passive: true });
    canvas.addEventListener('pointerdown', this.handlePointerDown, { passive: true });
    canvas.addEventListener('contextmenu', this.handleContextMenu);
    canvas.addEventListener('pointerenter', () => { this.isActive = true; });
    canvas.addEventListener('pointerleave', () => { this.isActive = false; });
    canvas.addEventListener('focusin', () => { this.isActive = true; });
    canvas.addEventListener('focusout', () => { this.isActive = false; });
    // Prevent page scroll when zooming over canvas
    canvas.addEventListener('wheel', this.handleWheelPreventDefault, { passive: false });
    canvas.addEventListener('pointerup', this.handlePointerUp, { passive: true });
    window.addEventListener('keydown', this.handleKeyDown);
    window.addEventListener('keyup', this.handleKeyUp);
    // Observe size changes to ensure canvas always fits
    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.handleResize());
      this.resizeObserver.observe(canvas);
    }
    // If initial size is zero, reflow next frame
    if (height === 0 || width === 0) {
      requestAnimationFrame(() => this.handleResize());
    }
  }

  add(object: THREE.Object3D, interactive = false): void {
    this.scene?.add(object);
    if (interactive) {
      this.interactionTargets.push(object);
    }
  }

  remove(object: THREE.Object3D): void {
    this.scene?.remove(object);
    this.interactionTargets = this.interactionTargets.filter((o) => o !== object);
  }

  onHover(callback: (intersections: THREE.Intersection[]) => void): void {
    this._onHover = callback;
  }
  onClick(callback: (intersections: THREE.Intersection[]) => void): void {
    this._onClick = callback;
  }
  onContextClick(callback: (intersections: THREE.Intersection[]) => void): void {
    this._onContextClick = callback;
  }
  private _onHover?: (intersections: THREE.Intersection[]) => void;
  private _onClick?: (intersections: THREE.Intersection[]) => void;
  private _onContextClick?: (intersections: THREE.Intersection[]) => void;

  start(): void {
    const tick = () => {
      this.animationHandle = requestAnimationFrame(tick);
      this.updateMovement();
      this.controls?.update();
      this.renderer?.render(this.scene!, this.camera!);
    };
    tick();
  }

  dispose(): void {
    cancelAnimationFrame(this.animationHandle);
    window.removeEventListener('resize', this.handleResize);
    this.canvas?.removeEventListener('pointermove', this.handlePointerMove);
    this.canvas?.removeEventListener('pointerdown', this.handlePointerDown);
    this.canvas?.removeEventListener('contextmenu', this.handleContextMenu);
    this.canvas?.removeEventListener('wheel', this.handleWheelPreventDefault as any);
    this.canvas?.removeEventListener('pointerup', this.handlePointerUp as any);
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('keyup', this.handleKeyUp);
    this.controls?.dispose();
    this.renderer?.dispose();
    this.interactionTargets = [];
    this.resizeObserver?.disconnect();
  }

  private handleResize = (): void => {
    if (!this.canvas || !this.camera || !this.renderer) return;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    const aspect = width / height;
    const halfH = (this.camera.top - this.camera.bottom) / 2;
    const halfW = halfH * aspect;
    this.camera.left = -halfW;
    this.camera.right = halfW;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  };

  private handlePointerMove = (event: PointerEvent): void => {
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycast('hover');
  };

  private handlePointerDown = (event: PointerEvent): void => {
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycast('click');
    if (this._onPointerDown) {
      const hits = this.computeHits();
      this._onPointerDown(hits, event.button);
    }
  };

  private handleContextMenu = (event: MouseEvent): void => {
    event.preventDefault();
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycast('context');
  };

  private handlePointerUp = (event: PointerEvent): void => {
    if (!this.renderer || !this.camera) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    if (this._onPointerUp) {
      const hits = this.computeHits();
      this._onPointerUp(hits, event.button);
    }
  };

  private handleWheelPreventDefault = (event: WheelEvent): void => {
    event.preventDefault();
  };

  private raycast(kind: 'hover' | 'click' | 'context'): void {
    if (!this.camera || !this.scene) return;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactionTargets, true);
    if (kind === 'hover' && this._onHover) this._onHover(hits);
    if (kind === 'click' && this._onClick) this._onClick(hits);
    if (kind === 'context' && this._onContextClick) this._onContextClick(hits);
  }

  private computeHits(): THREE.Intersection[] {
    if (!this.camera || !this.scene) return [];
    this.raycaster.setFromCamera(this.pointer, this.camera);
    return this.raycaster.intersectObjects(this.interactionTargets, true);
  }

  recenterTo(point: THREE.Vector3): void {
    if (!this.camera || !this.controls) return;
    const y = this.camera.position.y;
    this.controls.target.set(point.x, 0, point.z);
    this.camera.position.set(point.x, y, point.z);
  }

  fitToBounds(widthWorld: number, heightWorld: number, margin = 1.1): void {
    if (!this.camera) return;
    const baseWidth = this.camera.right - this.camera.left;
    const baseHeight = this.camera.top - this.camera.bottom;
    const zoomW = baseWidth / (widthWorld * margin);
    const zoomH = baseHeight / (heightWorld * margin);
    const zoom = Math.min(zoomW, zoomH);
    this.camera.zoom = Math.max(this.controls?.minZoom ?? 0.1, Math.min(zoom, this.controls?.maxZoom ?? 10));
    this.camera.updateProjectionMatrix();
  }

  worldToCanvas(point: THREE.Vector3): { x: number; y: number } | null {
    if (!this.camera || !this.renderer) return null;
    const p = point.clone().project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = (p.x * 0.5 + 0.5) * rect.width + rect.left;
    const y = (-p.y * 0.5 + 0.5) * rect.height + rect.top;
    return { x, y };
  }

  private handleKeyDown = (e: KeyboardEvent): void => {
    if (!this.isActive) return;
    this.pressed.add(e.code);
  };
  private handleKeyUp = (e: KeyboardEvent): void => {
    if (!this.isActive) return;
    this.pressed.delete(e.code);
  };
  private updateMovement(): void {
    if (!this.camera || !this.controls || !this.isActive) return;
    const speed = 0.6 * (1 / (this.camera.zoom || 1));
    let dx = 0;
    let dz = 0;
    if (this.pressed.has('KeyW')) dz -= speed;
    if (this.pressed.has('KeyS')) dz += speed;
    if (this.pressed.has('KeyA')) dx -= speed;
    if (this.pressed.has('KeyD')) dx += speed;
    if (dx !== 0 || dz !== 0) {
      this.controls.target.x += dx;
      this.controls.target.z += dz;
      this.camera.position.x += dx;
      this.camera.position.z += dz;
    }
  }

  onPointerDown(callback: (intersections: THREE.Intersection[], button: number) => void): void {
    this._onPointerDown = callback;
  }
  onPointerUp(callback: (intersections: THREE.Intersection[], button: number) => void): void {
    this._onPointerUp = callback;
  }
}


