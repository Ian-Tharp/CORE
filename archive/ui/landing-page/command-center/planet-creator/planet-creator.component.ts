import {
  Component, EventEmitter, Input, Output, OnChanges, OnDestroy, SimpleChanges, inject,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subject, switchMap, takeUntil } from 'rxjs';
import {
  WorldGenParams, BiomeArchetype, clampParams, planetLoreContext, planetArtClause,
} from '../engine/planet';
import { WorldsService } from '../../../services/worlds/worlds.service';
import { CreativeService } from '../../../services/creative/creative.service';

/**
 * Per-world planet creator — the controlled panel shown at PLANET altitude.
 * Edits emit (paramsChange) for the parent to re-apply (updatePlanet) + persist.
 * Hosts the "regenerate art / lore from this planet" actions, which fold the
 * planet's descriptor into the existing generate endpoints.
 */
@Component({
  selector: 'app-planet-creator',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './planet-creator.component.html',
  styleUrl: './planet-creator.component.scss',
})
export class PlanetCreatorComponent implements OnChanges, OnDestroy {
  /** Live params for the mounted planet (source of truth = parent). */
  @Input({ required: true }) params!: WorldGenParams;
  /** The backing world id; AI/persist actions are disabled until the world is saved. */
  @Input() worldId: string | null = null;
  /** Which orb these params belong to (per-orb persistence + asset tagging). */
  @Input() tileIndex = 0;

  /** Emitted (rAF-debounced) on every edit; the parent re-applies + persists. */
  @Output() paramsChange = new EventEmitter<WorldGenParams>();
  /** Notify the parent that art/lore was (re)generated, so it can reload. */
  @Output() regenerated = new EventEmitter<'art' | 'lore'>();

  private readonly worlds = inject(WorldsService);
  private readonly creative = inject(CreativeService);
  private readonly destroy$ = new Subject<void>();

  readonly archetypes: BiomeArchetype[] =
    ['temperate', 'desert', 'tundra', 'oceanic', 'volcanic', 'jungle', 'astral'];

  /** Local editable draft — ngModel needs a mutable object we own. */
  draft!: WorldGenParams;
  collapsed = false;
  busyArt = false;
  busyLore = false;
  status = '';
  private raf = 0;

  private readonly ART_QUALITY =
    'Ultra-detailed digital matte painting, volumetric lighting, sweeping planetary vista, '
    + 'luminous solarpunk science-fantasy, sharp focus, concept-art quality.';

  ngOnChanges(c: SimpleChanges): void {
    if (c['params'] && this.params) { this.draft = structuredClone(this.params); }
  }
  ngOnDestroy(): void { this.destroy$.next(); this.destroy$.complete(); }

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

  /** Generate world art grounded in this planet's params, persist it to the world. */
  regenerateArt(): void {
    if (!this.worldId || this.busyArt) { return; }
    this.busyArt = true; this.status = 'Painting this world…';
    const p = this.draft;
    const prompt = `A world named "${p.seed}". ${planetArtClause(p)} ${this.ART_QUALITY}`;
    this.creative.generateImage(prompt).pipe(
      switchMap(({ b64 }) => this.worlds.saveAsset(this.worldId!, {
        image_b64: b64, kind: 'art',
        title: `${p.seed} — ${p.biomeArchetype} portrait`,
        tile_index: this.tileIndex,
      })),
      takeUntil(this.destroy$),
    ).subscribe({
      next: () => { this.busyArt = false; this.status = 'Art saved ✓'; this.regenerated.emit('art'); },
      error: () => { this.busyArt = false; this.status = 'Art generation failed.'; },
    });
  }

  /** Generate an Overview lore page grounded in this planet's params. */
  regenerateLore(): void {
    if (!this.worldId || this.busyLore) { return; }
    this.busyLore = true; this.status = 'Writing this world’s lore…';
    const p = this.draft;
    this.worlds.generateLore(this.worldId, {
      kind: 'Overview',
      focus: 'this planet — its lands, climate, peoples and character',
      world_name: p.seed,
      context: planetLoreContext(p),
    }).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => { this.busyLore = false; this.status = 'Lore saved ✓'; this.regenerated.emit('lore'); },
      error: () => { this.busyLore = false; this.status = 'Lore generation failed.'; },
    });
  }
}
