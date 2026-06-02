import { Component, ElementRef, NgZone, OnDestroy, AfterViewInit, ViewChild, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildPlanet, updatePlanet, disposePlanet } from '../planet-factory';
import { WorldGenParams, DEFAULT_PLANET_PARAMS, BiomeArchetype, LodTier } from '../world-gen-params';

/** Standalone tuning lab for the procedural planet module (/command-center/planet-lab). */
@Component({
  selector: 'app-planet-lab',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './planet-lab.component.html',
  styleUrl: './planet-lab.component.scss',
})
export class PlanetLabComponent implements AfterViewInit, OnDestroy {
  @ViewChild('host', { static: true }) host!: ElementRef<HTMLDivElement>;
  private readonly zone = inject(NgZone);

  readonly archetypes: BiomeArchetype[] = ['temperate', 'desert', 'tundra', 'oceanic', 'volcanic', 'jungle', 'astral'];
  readonly tiers: LodTier[] = ['preview', 'standard', 'high', 'ultra'];
  params: WorldGenParams = structuredClone(DEFAULT_PLANET_PARAMS);
  tier = signal<LodTier>('standard');
  readonly vertexCountSig = signal(0);

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private planet!: THREE.Group;
  private raf = 0;
  private debounce = 0;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  ngAfterViewInit(): void {
    const el = this.host.nativeElement;
    const w = el.clientWidth || 1, h = el.clientHeight || 1;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    this.camera.position.set(0, 0.6, 3);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h, false);
    this.renderer.shadowMap.enabled = true;
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.6;

    this.scene.add(new THREE.HemisphereLight(0xdffcff, 0x081410, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(3, 2, 2);
    dir.castShadow = true;
    this.scene.add(dir);

    this.planet = buildPlanet(this.params, { tier: this.tier() });
    this.scene.add(this.planet);
    this.vertexCountSig.set((this.planet.userData as { vertexCount: number }).vertexCount);

    this.zone.runOutsideAngular(() => {
      const tick = () => {
        this.raf = requestAnimationFrame(tick);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      };
      tick();
    });
    addEventListener('resize', this.onResize);
  }

  /** Debounced to one rAF so rebuild-class changes never run inside pointermove. */
  onParamsChange(): void {
    cancelAnimationFrame(this.debounce);
    this.debounce = requestAnimationFrame(() => {
      updatePlanet(this.planet, this.params, { tier: this.tier() });
      this.vertexCountSig.set((this.planet.userData as { vertexCount: number }).vertexCount);
    });
  }

  onTierChange(t: LodTier): void { this.tier.set(t); this.onParamsChange(); }
  randomizeSeed(): void { this.params = { ...this.params, seed: Math.random().toString(36).slice(2, 10) }; this.onParamsChange(); }

  private onResize = (): void => {
    const el = this.host.nativeElement;
    this.camera.aspect = (el.clientWidth || 1) / (el.clientHeight || 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(el.clientWidth, el.clientHeight, false);
  };

  ngOnDestroy(): void {
    cancelAnimationFrame(this.raf);
    cancelAnimationFrame(this.debounce);
    removeEventListener('resize', this.onResize);
    if (this.planet) { disposePlanet(this.planet); }
    this.controls?.dispose();
    this.renderer?.dispose();
  }
}
