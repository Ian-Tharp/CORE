import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { ProjectService, HexWorldSnapshot } from '../../landing-page/command-center/engine/project.service';

@Component({
  selector: 'app-world-detail',
  imports: [CommonModule, MatButtonModule],
  templateUrl: './world-detail.component.html',
  styleUrl: './world-detail.component.scss'
})
export class WorldDetailComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly projects = inject(ProjectService);
  world?: HexWorldSnapshot;
  /** True once the load has been attempted — drives the not-found state. */
  loaded = false;

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id') ?? '';
    this.world = id ? this.projects.load(id) : undefined;
    this.loaded = true;
  }

  openInEditor(): void {
    if (!this.world) {return;}
    this.router.navigate(['/command-center/edit'], { queryParams: { projectId: this.world.id } });
  }

  openBoards(): void {
    if (!this.world) {return;}
    this.router.navigate(['/creative/boards'], { queryParams: { projectId: this.world.id } });
  }

  openWiki(): void {
    if (!this.world) {return;}
    this.router.navigate(['/creative/wiki'], { queryParams: { projectId: this.world.id } });
  }

  backToWorlds(): void {
    this.router.navigate(['/creative/worlds']);
  }
}
