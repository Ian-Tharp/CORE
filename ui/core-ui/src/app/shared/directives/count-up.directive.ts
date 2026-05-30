import {
  Directive,
  ElementRef,
  Input,
  NgZone,
  OnChanges,
  OnDestroy,
  inject
} from '@angular/core';

/**
 * Tweens an element's text content from its previous value up (or down) to the
 * bound target number using `requestAnimationFrame`.
 *
 * The animation runs OUTSIDE Angular (via {@link NgZone.runOutsideAngular}) so
 * it never triggers change detection per frame, and it renders the final value
 * synchronously when the user prefers reduced motion.
 *
 * Usage: `<span [appCountUp]="value" [countUpDuration]="800"></span>`
 */
@Directive({
  selector: '[appCountUp]',
  standalone: true
})
export class CountUpDirective implements OnChanges, OnDestroy {
  private readonly _el = inject(ElementRef) as ElementRef<HTMLElement>;
  private readonly _zone = inject(NgZone);

  /** Target value to animate toward. */
  @Input('appCountUp') public target = 0;

  /** Animation duration in milliseconds. */
  @Input() public countUpDuration = 800;

  /** Number of decimal places to render. */
  @Input() public countUpDecimals = 0;

  private _current = 0;
  private _rafId: number | null = null;

  public ngOnChanges(): void {
    const to = Number(this.target) || 0;
    if (this._prefersReducedMotion()) {
      this._cancel();
      this._render(to);
      this._current = to;
      return;
    }
    this._animate(this._current, to);
  }

  public ngOnDestroy(): void {
    this._cancel();
  }

  private _animate(from: number, to: number): void {
    this._cancel();

    const duration = Math.max(0, this.countUpDuration);
    if (from === to || duration === 0 || typeof requestAnimationFrame === 'undefined') {
      this._render(to);
      this._current = to;
      return;
    }

    this._zone.runOutsideAngular(() => {
      let start: number | null = null;
      const step = (ts: number) => {
        if (start === null) {
          start = ts;
        }
        const t = Math.min(1, (ts - start) / duration);
        // easeOutCubic for a snappy settle.
        const eased = 1 - Math.pow(1 - t, 3);
        this._render(from + (to - from) * eased);
        if (t < 1) {
          this._rafId = requestAnimationFrame(step);
        } else {
          this._current = to;
          this._render(to);
          this._rafId = null;
        }
      };
      this._rafId = requestAnimationFrame(step);
    });
  }

  private _render(value: number): void {
    const factor = Math.pow(10, this.countUpDecimals);
    const rounded = Math.round(value * factor) / factor;
    this._el.nativeElement.textContent = rounded.toFixed(this.countUpDecimals);
  }

  private _cancel(): void {
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  private _prefersReducedMotion(): boolean {
    return (
      typeof matchMedia !== 'undefined' &&
      matchMedia('(prefers-reduced-motion: reduce)').matches
    );
  }
}
