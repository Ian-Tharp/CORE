import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { CountUpDirective } from '../../../shared/directives/count-up.directive';

/**
 * A compact radial gauge for a single system vital (CPU, memory, etc.).
 * The arc fills proportionally to `value` (0-100) and shifts colour as it
 * approaches saturation to read as a HUD warning.
 */
@Component({
  selector: 'app-vitals-ring',
  standalone: true,
  imports: [CountUpDirective],
  templateUrl: './vitals-ring.component.html',
  styleUrl: './vitals-ring.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class VitalsRingComponent {
  public readonly label = input<string>('');
  public readonly detail = input<string>('');
  public readonly value = input<number>(0);

  public readonly radius = 42;
  public readonly circumference = 2 * Math.PI * 42;

  /** Clamped 0-100 value used for rendering. */
  public readonly clamped = computed<number>(() =>
    Math.min(100, Math.max(0, Math.round(this.value())))
  );

  /** Dash offset that draws the arc to the current value. */
  public readonly dashOffset = computed<number>(() => {
    return this.circumference * (1 - this.clamped() / 100);
  });

  /** Severity bucket used for colour theming. */
  public readonly severity = computed<'nominal' | 'elevated' | 'critical'>(() => {
    const v = this.clamped();
    if (v >= 85) {
      return 'critical';
    }
    if (v >= 65) {
      return 'elevated';
    }
    return 'nominal';
  });
}
