import { Injectable, inject } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({ providedIn: 'root' })
export class UiNotifyService {
  private readonly snack = inject(MatSnackBar);

  showError(message: string, action = 'Dismiss', durationMs = 5000): void {
    this.snack.open(message, action, { duration: durationMs, panelClass: ['mat-warn'] });
  }

  showInfo(message: string, action = 'OK', durationMs = 3000): void {
    this.snack.open(message, action, { duration: durationMs });
  }

  showSuccess(message: string, action = 'Nice', durationMs = 3000): void {
    this.snack.open(message, action, { duration: durationMs });
  }
}


