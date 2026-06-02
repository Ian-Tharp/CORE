import { Component, EventEmitter, Input, Output, OnChanges, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { WorldGenParams, BiomeArchetype, clampParams } from '../engine/planet';

/**
 * Per-world planet creator — a PURE controlled panel shown at PLANET altitude.
 * It owns no THREE.js/renderer (the command center owns the mounted planet);
 * edits emit (paramsChange) and the parent re-applies via updatePlanet + persists.
 */
@Component({
  selector: 'app-planet-creator',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './planet-creator.component.html',
  styleUrl: './planet-creator.component.scss',
})
export class PlanetCreatorComponent implements OnChanges {
  /** Live params for the mounted planet (source of truth = parent). */
  @Input({ required: true }) params!: WorldGenParams;
  /** The backing world id; AI/persist actions are disabled until the world is saved. */
  @Input() worldId: string | null = null;

  /** Emitted (rAF-debounced) on every edit; the parent re-applies + persists. */
  @Output() paramsChange = new EventEmitter<WorldGenParams>();
  /** Ask the parent to regenerate art/lore from the current params. */
  @Output() regenerate = new EventEmitter<'art' | 'lore'>();

  readonly archetypes: BiomeArchetype[] =
    ['temperate', 'desert', 'tundra', 'oceanic', 'volcanic', 'jungle', 'astral'];

  /** Local editable draft — ngModel needs a mutable object we own. */
  draft!: WorldGenParams;
  collapsed = false;
  private raf = 0;

  ngOnChanges(c: SimpleChanges): void {
    if (c['params'] && this.params) { this.draft = structuredClone(this.params); }
  }

  /** Debounced to one rAF so rebuild-class edits never fire mid-pointermove. */
  onChange(): void {
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() =>
      this.paramsChange.emit(clampParams(structuredClone(this.draft))));
  }

  randomizeSeed(): void {
    this.draft = { ...this.draft, seed: Math.random().toString(36).slice(2, 10) };
    this.onChange();
  }

  toggleCollapsed(): void { this.collapsed = !this.collapsed; }
}
