import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { ProjectService, HexWorldSnapshot } from '../../landing-page/command-center/engine/project.service';

@Component({
  selector: 'app-world-detail',
  imports: [CommonModule],
  templateUrl: './world-detail.component.html',
  styleUrl: './world-detail.component.scss'
})
export class WorldDetailComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly projects = inject(ProjectService);
  world?: HexWorldSnapshot;

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id') ?? '';
    this.world = this.projects.load(id);
  }

  openInEditor(): void {
    if (!this.world) return;
    this.router.navigate(['/command-center'], { queryParams: { projectId: this.world.id } });
  }

  openBoards(): void {
    if (!this.world) return;
    this.router.navigate(['/creative/boards'], { queryParams: { projectId: this.world.id } });
  }

  openWiki(): void {
    if (!this.world) return;
    this.router.navigate(['/creative/wiki'], { queryParams: { projectId: this.world.id } });
  }
}
