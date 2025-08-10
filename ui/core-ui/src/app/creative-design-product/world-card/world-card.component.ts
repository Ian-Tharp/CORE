import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import type { HexWorldSnapshot } from '../../landing-page/command-center/engine/project.service';

@Component({
  selector: 'app-world-card',
  imports: [CommonModule, RouterLink],
  templateUrl: './world-card.component.html',
  styleUrl: './world-card.component.scss'
})
export class WorldCardComponent {
  @Input({ required: true }) world!: HexWorldSnapshot;
}
