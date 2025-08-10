import { Component, OnInit, OnDestroy } from '@angular/core';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatBadgeModule } from '@angular/material/badge';
import { Router, NavigationEnd, ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { filter, takeUntil } from 'rxjs/operators';

interface Breadcrumb {
  label: string;
  url: string;
  isActive: boolean;
}

@Component({
  selector: 'app-top-navigation',
  imports: [
    CommonModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatBadgeModule
  ],
  templateUrl: './top-navigation.component.html',
  styleUrl: './top-navigation.component.scss'
})
export class TopNavigationComponent implements OnInit, OnDestroy {
  systemStatus = 'operational';
  agentCount = 3;
  activeWorkflows = 2;
  
  breadcrumbs: Breadcrumb[] = [];
  private destroy$ = new Subject<void>();

  constructor(
    private router: Router,
    private activatedRoute: ActivatedRoute,
    private location: Location
  ) {}

  ngOnInit(): void {
    // Build initial breadcrumbs
    this.buildBreadcrumbs();
    
    // Listen to route changes
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd),
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.buildBreadcrumbs();
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private buildBreadcrumbs(): void {
    const url = this.router.url;
    const segments = url.split('/').filter(segment => segment);
    
    // Start with home/landing page
    this.breadcrumbs = [{
      label: 'Landing Page',
      url: '/',
      isActive: segments.length === 0
    }];
    
    // Build breadcrumb for each segment
    let currentUrl = '';
    segments.forEach((segment, index) => {
      currentUrl += `/${segment}`;
      const isLast = index === segments.length - 1;
      
      this.breadcrumbs.push({
        label: this.getLabelForSegment(segment),
        url: currentUrl,
        isActive: isLast
      });
    });
    
    // Remove any accidental duplicates while preserving order
    const seen = new Set<string>();
    this.breadcrumbs = this.breadcrumbs.filter(b => {
      const key = `${b.label}|${b.url}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  private getLabelForSegment(segment: string): string {
    // Map route segments to display labels
    const labelMap: { [key: string]: string } = {
      'conversations': 'Conversations',
      'knowledge': 'Knowledge Base',
      'knowledgebase': 'Knowledge Base',
      'knowledge-base': 'Knowledge Base',
      'command-center': 'Command Center',
      'agents': 'Agents',
      'workflows': 'Workflows',
      'settings': 'Settings'
    };
    
    return labelMap[segment.toLowerCase()] || 
           segment.split('-').map(word => 
             word.charAt(0).toUpperCase() + word.slice(1)
           ).join(' ');
  }

  navigateTo(url: string): void {
    this.router.navigate([url]);
  }

  goBack(): void {
    this.location.back();
  }
}
