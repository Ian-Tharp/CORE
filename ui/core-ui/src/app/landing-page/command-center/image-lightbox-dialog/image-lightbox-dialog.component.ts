import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';

/** A focused, full-scale view of a generated world image / portrait. */
@Component({
  selector: 'app-image-lightbox-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule],
  template: `
    <figure class="lightbox">
      <button class="lightbox__close" (click)="ref.close()" aria-label="Close">&#10005;</button>
      <img class="lightbox__img" [src]="data.url" [alt]="data.title || 'Generated image'" />
      @if (data.title) {
        <figcaption class="lightbox__cap">{{ data.title }}</figcaption>
      }
    </figure>
  `,
  styleUrl: './image-lightbox-dialog.component.scss'
})
export class ImageLightboxDialogComponent {
  constructor(
    public readonly ref: MatDialogRef<ImageLightboxDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { url: string; title?: string }
  ) {}
}
