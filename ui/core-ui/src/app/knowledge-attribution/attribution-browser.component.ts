import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { Observable } from 'rxjs';

import { KnowledgebaseService } from '../services/knowledgebase/knowledgebase.service';
import {
  AttributedKnowledge,
  AttributionFilter,
  AttributionStats,
  AttributionVerificationStatus,
  KnowledgeAttribution
} from '../models/attribution.models';

@Component({
  selector: 'app-attribution-browser',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './attribution-browser.component.html',
  styleUrls: ['./attribution-browser.component.scss']
})
export class AttributionBrowserComponent implements OnInit {
  private knowledgebaseService = inject(KnowledgebaseService);

  // Data
  attributedKnowledge$: Observable<AttributedKnowledge[]> = new Observable();
  attributionStats$: Observable<AttributionStats> = new Observable();
  loading = false;
  error: string | null = null;

  // Filters
  currentFilter: AttributionFilter = {};
  searchQuery = '';
  selectedStatuses: AttributionVerificationStatus[] = [];
  selectedInstanceId = '';
  dateRangeStart = '';
  dateRangeEnd = '';

  // Enums for template
  AttributionVerificationStatus = AttributionVerificationStatus;
  verificationStatusOptions = [
    { value: AttributionVerificationStatus.VERIFIED, label: 'Verified' },
    { value: AttributionVerificationStatus.PENDING, label: 'Pending' },
    { value: AttributionVerificationStatus.FAILED, label: 'Failed' },
    { value: AttributionVerificationStatus.UNVERIFIED, label: 'Unverified' }
  ];

  ngOnInit(): void {
    this.loadData();
  }

  private loadData(): void {
    this.loading = true;
    this.error = null;

    // Load stats
    this.attributionStats$ = this.knowledgebaseService.getAttributionStats();

    // Load attributed knowledge with current filter
    this.attributedKnowledge$ = this.knowledgebaseService.getAttributedKnowledge(this.currentFilter);

    // Simulate loading complete (in real app, this would be handled by the observables)
    setTimeout(() => {
      this.loading = false;
    }, 500);
  }

  applyFilters(): void {
    this.currentFilter = {
      searchQuery: this.searchQuery || undefined,
      instanceId: this.selectedInstanceId || undefined,
      verificationStatus: this.selectedStatuses.length > 0 ? this.selectedStatuses : undefined,
      dateRange: this.dateRangeStart && this.dateRangeEnd ? {
        start: new Date(this.dateRangeStart),
        end: new Date(this.dateRangeEnd)
      } : undefined
    };

    this.loadData();
  }

  clearFilters(): void {
    this.searchQuery = '';
    this.selectedStatuses = [];
    this.selectedInstanceId = '';
    this.dateRangeStart = '';
    this.dateRangeEnd = '';
    this.currentFilter = {};
    this.loadData();
  }

  verifyAttribution(attribution: KnowledgeAttribution): void {
    this.knowledgebaseService.verifyAttribution(attribution.id).subscribe({
      next: (updatedAttribution) => {
        // In a real implementation, we'd update the local data
        console.log('Attribution verified:', updatedAttribution);
        this.loadData(); // Refresh data
      },
      error: (error) => {
        console.error('Failed to verify attribution:', error);
        this.error = 'Failed to verify attribution';
      }
    });
  }

  getStatusBadgeClass(status: AttributionVerificationStatus): string {
    switch (status) {
      case AttributionVerificationStatus.VERIFIED:
        return 'badge-success';
      case AttributionVerificationStatus.PENDING:
        return 'badge-warning';
      case AttributionVerificationStatus.FAILED:
        return 'badge-error';
      case AttributionVerificationStatus.UNVERIFIED:
        return 'badge-neutral';
      default:
        return 'badge-neutral';
    }
  }

  formatTimestamp(date: Date): string {
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  }

  truncateText(text: string, maxLength: number = 200): string {
    if (text.length <= maxLength) {return text;}
    return text.substring(0, maxLength) + '...';
  }

  trackByKnowledgeId(index: number, item: AttributedKnowledge): string {
    return item.id;
  }
}