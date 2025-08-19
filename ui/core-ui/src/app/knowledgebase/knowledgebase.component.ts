import { Component, OnInit, OnDestroy, ViewChild, ElementRef, inject, TemplateRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormControl, FormGroup, FormBuilder } from '@angular/forms';
import { MatTabsModule } from '@angular/material/tabs';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatMenuModule } from '@angular/material/menu';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatBadgeModule } from '@angular/material/badge';
import { MatDividerModule } from '@angular/material/divider';
import { MatTableModule } from '@angular/material/table';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { CdkDragDrop, DragDropModule, moveItemInArray } from '@angular/cdk/drag-drop';
import { Subject, BehaviorSubject, takeUntil, debounceTime, distinctUntilChanged, combineLatest, map, startWith, firstValueFrom, Observable } from 'rxjs';
import { trigger, state, style, transition, animate, stagger, query } from '@angular/animations';

import { KnowledgebaseService } from '../services/knowledgebase/knowledgebase.service';
import {
  KnowledgeFile,
  FileTag,
  KnowledgebaseFilter,
  FileSource,
  FileStatus,
  EmbeddingStats,
  ActivityLog
} from '../models/knowledgebase.models';

type ViewMode = 'grid' | 'list' | 'timeline';
type TabView = 'personal' | 'global' | 'activity';

@Component({
  selector: 'app-knowledgebase',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatTabsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatMenuModule,
    MatTooltipModule,
    MatExpansionModule,
    MatDialogModule,
    MatSnackBarModule,
    MatAutocompleteModule,
    MatBadgeModule,
    MatDividerModule,
    MatTableModule,
    MatCheckboxModule,
    MatPaginatorModule,
    DragDropModule
  ],
  templateUrl: './knowledgebase.component.html',
  styleUrl: './knowledgebase.component.scss',
  animations: [
    trigger('fileAnimation', [
      transition('* => *', [
        query(':enter', [
          style({ opacity: 0, transform: 'translateY(20px)' }),
          stagger(50, [
            animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
          ])
        ], { optional: true })
      ])
    ]),
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('300ms ease-in', style({ opacity: 1 }))
      ])
    ])
  ]
})
export class KnowledgebaseComponent implements OnInit, OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  @ViewChild('semanticSearchDialog') semanticSearchDialog!: TemplateRef<any>;
  
  private destroy$ = new Subject<void>();
  private fb = inject(FormBuilder);
  private dialog = inject(MatDialog);
  private snackBar = inject(MatSnackBar);
  knowledgebaseService = inject(KnowledgebaseService);

  // State
  currentTab: TabView = 'personal';
  viewMode: ViewMode = 'grid';
  selectedFiles = new Set<string>();
  isLoading = false;
  uploadProgress: Record<string, number> = {};
  uploadStage: Record<string, string> = {};

  // Helper for template to list active upload names
  get uploadingNames(): string[] {
    return Object.keys(this.uploadProgress);
  }

  // Form controls - initialize immediately
  searchControl = new FormControl('');
  filterForm!: FormGroup;
  tagInput = new FormControl('');

  // Data streams - use the service directly
  files$ = this.knowledgebaseService.files$;
  stats$ = this.knowledgebaseService.stats$;
  totalBytes$ = this.files$.pipe(map(files => files.reduce((sum, f) => sum + (f.size || 0), 0)));
  totalFiles$ = this.files$.pipe(map(files => files.length));
  availableTags$ = this.knowledgebaseService.availableTags$;
  private tagIdToName: Record<string, string> = {};
  recentActivity: ActivityLog[] = [];

  // Filtered files observable - will be initialized in setupFilteredFiles()
  filteredFiles$!: Observable<KnowledgeFile[]>;
  // Grid pagination
  private pageIndex$ = new BehaviorSubject<number>(0);
  private pageSize$ = new BehaviorSubject<number>(12);
  pagedFiles$!: Observable<KnowledgeFile[]>;
  gridPageIndex = 0;
  gridPageSize = 12;
  gridPageSizeOptions: number[] = [8, 12, 16, 24];

  // Enums for template
  FileSource = FileSource;
  FileStatus = FileStatus;

  // File type icons mapping
  fileTypeIcons: Record<string, string> = {
    'application/pdf': 'picture_as_pdf',
    'text/plain': 'description',
    'text/markdown': 'description',
    'application/json': 'data_object',
    'text/csv': 'table_chart',
    'image/': 'image',
    'video/': 'video_file',
    'audio/': 'audio_file',
    'application/zip': 'folder_zip',
    'default': 'insert_drive_file'
  };

  // Source colors
  sourceColors: Record<FileSource, string> = {
    [FileSource.USER_UPLOAD]: '#4caf50',
    [FileSource.CONVERSATION_HISTORY]: '#2196f3',
    [FileSource.WORKFLOW_ARTIFACT]: '#ff9800',
    [FileSource.AGENT_GENERATED]: '#9c27b0',
    [FileSource.SYSTEM_IMPORT]: '#607d8b',
    [FileSource.WEB_SCRAPE]: '#00bcd4'
  };

  constructor() {
    // Initialize form in constructor
    this.filterForm = this.fb.group({
      tags: [[]],
      fileTypes: [[]],
      sources: [[]],
      dateRange: this.fb.group({
        start: [null],
        end: [null]
      }),
      status: [[]]
    });
    
    // Initialize filtered files observable
    this.setupFilteredFiles();
  }

  private setupFilteredFiles(): void {
    // Filtered files based on current tab
    this.filteredFiles$ = combineLatest([
      this.files$,
      this.searchControl.valueChanges.pipe(startWith(''), debounceTime(300))
    ]).pipe(
      map(([files, searchTerm]) => {
        let filtered = files;
        
        // Filter by tab
        if (this.currentTab === 'personal') {
          filtered = filtered.filter(f => !f.isGlobal);
        } else if (this.currentTab === 'global') {
          filtered = filtered.filter(f => f.isGlobal);
        }
        
        // Apply search
        if (searchTerm) {
          const term = searchTerm.toLowerCase();
          filtered = filtered.filter(f =>
            (f.title?.toLowerCase?.().includes(term) ?? false) ||
            f.filename.toLowerCase().includes(term) ||
            f.description?.toLowerCase().includes(term) ||
            f.originalName.toLowerCase().includes(term)
          );
        }
        
        return filtered;
      })
    );

    // Derive paged files for grid view
    this.pagedFiles$ = combineLatest([
      this.filteredFiles$,
      this.pageIndex$,
      this.pageSize$
    ]).pipe(
      map(([files, index, size]) => {
        const start = index * size;
        return files.slice(start, start + size);
      })
    );
  }

  ngOnInit(): void {
    // Load persisted grid page size
    const savedSize = Number(localStorage.getItem('kb.gridPageSize') || '0');
    if (savedSize > 0) {
      this.gridPageSize = savedSize;
      // initialize the subject so first page uses saved size
      // Note: pageSize$ default is 12; update to saved value
      (this as any).pageSize$.next(savedSize);
    }
    // Subscribe to upload progress
    this.knowledgebaseService.uploadProgress$
      .pipe(takeUntil(this.destroy$))
      .subscribe(progress => {
        this.uploadProgress[progress.fileId] = progress.progress;
      });

    // Subscribe to upload stage updates
    this.knowledgebaseService.uploadStage$
      .pipe(takeUntil(this.destroy$))
      .subscribe(stage => {
        this.uploadStage[stage.fileId] = stage.stage;
      });

    // Load recent activity
    this.loadRecentActivity();

    // Maintain a lookup for tag names for active filter chips
    this.availableTags$
      .pipe(takeUntil(this.destroy$))
      .subscribe(tags => {
        const map: Record<string, string> = {};
        for (const t of tags) { map[t.id] = t.name; }
        this.tagIdToName = map;
      });

    // Reset pagination on search changes
    this.searchControl.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.resetGridPagination());

    // Subscribe to filter changes
    this.filterForm.valueChanges
      .pipe(
        takeUntil(this.destroy$),
        debounceTime(500),
        distinctUntilChanged()
      )
      .subscribe(filters => {
        this.applyFilters();
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // Tab navigation
  onTabChange(tab: TabView): void {
    this.currentTab = tab;
    this.clearSelection();
    if (tab === 'activity') {
      this.loadRecentActivity();
    }
    this.resetGridPagination();
  }

  // View mode
  toggleViewMode(mode: ViewMode): void {
    this.viewMode = mode;
  }

  // File operations
  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      Array.from(input.files).forEach(file => this.uploadFile(file));
    }
  }

  uploadFile(file: File, isGlobal: boolean = this.currentTab === 'global'): void {
    const tags = this.tagInput.value ? [this.tagInput.value] : [];
    
    this.knowledgebaseService.uploadFile({
      file,
      isGlobal,
      tags,
      processImmediately: true
    }).subscribe({
      next: (result) => {
        if (result) {
          this.snackBar.open(`File "${file.name}" uploaded successfully`, 'Close', {
            duration: 3000
          });
          delete this.uploadProgress[file.name];
          delete this.uploadStage[file.name];
        }
      },
      error: (error) => {
        const msg = error?.status === 409 ? `Upload failed: "${file.name}" already exists in your knowledgebase` : `Failed to upload "${file.name}"`;
        this.snackBar.open(msg, 'Close', { duration: 5000 });
        delete this.uploadProgress[file.name];
        delete this.uploadStage[file.name];
      }
    });
  }

  deleteFile(file: KnowledgeFile): void {
    if (confirm(`Are you sure you want to delete "${file.filename}"?`)) {
      this.knowledgebaseService.deleteFile(file.id).subscribe({
        next: () => {
          this.snackBar.open('File deleted successfully', 'Close', {
            duration: 3000
          });
        },
        error: () => {
          this.snackBar.open('Failed to delete file', 'Close', {
            duration: 5000
          });
        }
      });
    }
  }

  processEmbeddings(file: KnowledgeFile): void {
    this.knowledgebaseService.processFileEmbeddings(file.id).subscribe({
      next: () => {
        this.snackBar.open('Processing started', 'Close', {
          duration: 3000
        });
      },
      error: () => {
        this.snackBar.open('Failed to start processing', 'Close', {
          duration: 5000
        });
      }
    });
  }

  onReextractTitle(file: KnowledgeFile): void {
    this.knowledgebaseService.reextractTitle(file.id).subscribe({
      next: (res) => {
        if (res.updated && res.title) {
          this.snackBar.open('Title updated', 'Close', { duration: 2500 });
        } else {
          this.snackBar.open('No better title found', 'Close', { duration: 2500 });
        }
      },
      error: () => {
        this.snackBar.open('Failed to re-extract title', 'Close', { duration: 4000 });
      }
    });
  }

  copyFilename(file: KnowledgeFile): void {
    try {
      navigator.clipboard.writeText(file.filename);
      this.snackBar.open('Filename copied', 'Close', { duration: 2000 });
    } catch {
      this.snackBar.open('Copy failed', 'Close', { duration: 3000 });
    }
  }

  // Selection
  toggleFileSelection(fileId: string): void {
    if (this.selectedFiles.has(fileId)) {
      this.selectedFiles.delete(fileId);
    } else {
      this.selectedFiles.add(fileId);
    }
  }

  selectAll(): void {
    this.filteredFiles$.pipe(takeUntil(this.destroy$)).subscribe(files => {
      files.forEach(f => this.selectedFiles.add(f.id));
    });
  }

  clearSelection(): void {
    this.selectedFiles.clear();
  }

  // Bulk operations
  deleteSelected(): void {
    if (this.selectedFiles.size === 0) return;
    
    if (confirm(`Delete ${this.selectedFiles.size} selected files?`)) {
      const deletions = Array.from(this.selectedFiles).map(id =>
        this.knowledgebaseService.deleteFile(id)
      );
      
      // Execute all deletions using firstValueFrom
      Promise.all(deletions.map(d => firstValueFrom(d))).then(() => {
        this.snackBar.open(`Deleted ${this.selectedFiles.size} files`, 'Close', {
          duration: 3000
        });
        this.clearSelection();
      });
    }
  }

  // Filtering
  applyFilters(): void {
    const filter: KnowledgebaseFilter = {
      ...this.filterForm.value,
      searchQuery: this.searchControl.value || undefined,
      isGlobal: this.currentTab === 'global' ? true : 
                this.currentTab === 'personal' ? false : undefined
    };
    
    this.knowledgebaseService.applyFilter(filter);
    this.resetGridPagination();
  }

  clearFilters(): void {
    this.filterForm.reset();
    this.searchControl.reset();
    this.knowledgebaseService.clearFilters();
  }

  // Active filter chips helpers
  get activeFilterChips(): Array<{ label: string; type: string; value: string }>{
    const chips: Array<{ label: string; type: string; value: string }> = [];
    const form = this.filterForm.value;

    // Tags
    for (const id of (form.tags as string[] || [])) {
      const name = this.tagIdToName[id] || id;
      chips.push({ label: `Tag: ${name}`, type: 'tag', value: id });
    }

    // File types
    for (const t of (form.fileTypes as string[] || [])) {
      chips.push({ label: `Type: ${t.split('/')[1] || t}`, type: 'type', value: t });
    }

    // Sources
    for (const s of (form.sources as FileSource[] || [])) {
      chips.push({ label: `Source: ${this.getSourceLabel(s)}`, type: 'source', value: s });
    }

    // Statuses
    for (const st of (form.status as FileStatus[] || [])) {
      chips.push({ label: `Status: ${st}`, type: 'status', value: st });
    }

    // Date range
    if (form.dateRange?.start || form.dateRange?.end) {
      const start = form.dateRange?.start ? new Date(form.dateRange.start).toLocaleDateString() : '...';
      const end = form.dateRange?.end ? new Date(form.dateRange.end).toLocaleDateString() : '...';
      chips.push({ label: `Date: ${start} – ${end}` , type: 'date', value: 'date' });
    }

    // Search
    if ((this.searchControl.value || '').trim()) {
      chips.push({ label: `Search: ${this.searchControl.value}` , type: 'search', value: 'search' });
    }

    return chips;
  }

  removeFilterChip(chip: { type: string; value: string }): void {
    const formVal = { ...this.filterForm.value } as any;
    switch (chip.type) {
      case 'tag':
        formVal.tags = (formVal.tags || []).filter((id: string) => id !== chip.value);
        break;
      case 'type':
        formVal.fileTypes = (formVal.fileTypes || []).filter((t: string) => t !== chip.value);
        break;
      case 'source':
        formVal.sources = (formVal.sources || []).filter((s: string) => s !== chip.value);
        break;
      case 'status':
        formVal.status = (formVal.status || []).filter((s: string) => s !== chip.value);
        break;
      case 'date':
        formVal.dateRange = { start: null, end: null };
        break;
      case 'search':
        this.searchControl.reset('');
        break;
    }
    this.filterForm.setValue(formVal);
    this.applyFilters();
  }

  // Pagination handlers for grid view
  onGridPageChange(event: PageEvent): void {
    this.gridPageIndex = event.pageIndex;
    this.gridPageSize = event.pageSize;
    this.pageIndex$.next(event.pageIndex);
    this.pageSize$.next(event.pageSize);
    try { localStorage.setItem('kb.gridPageSize', String(event.pageSize)); } catch {}
  }

  private resetGridPagination(): void {
    this.gridPageIndex = 0;
    this.pageIndex$.next(0);
  }

  // TrackBy for perf
  trackByFileId(index: number, file: KnowledgeFile): string {
    return file.id;
  }

  // Title helpers for tooltip/length
  getFullTitle(file: KnowledgeFile): string {
    return (file.title && file.title.trim()) || file.originalName || file.filename;
  }

  isTitleLong(file: KnowledgeFile, limit: number = 60): boolean {
    return this.getFullTitle(file).length > limit;
  }

  // Activity
  loadRecentActivity(): void {
    this.knowledgebaseService.getRecentActivity().subscribe(activities => {
      this.recentActivity = activities;
    });
  }

  // Helpers
  getFileIcon(mimeType: string): string {
    for (const [key, icon] of Object.entries(this.fileTypeIcons)) {
      if (mimeType.startsWith(key)) {
        return icon;
      }
    }
    return this.fileTypeIcons['default'];
  }

  getSourceLabel(source: FileSource): string {
    const labels: Record<FileSource, string> = {
      [FileSource.USER_UPLOAD]: 'User Upload',
      [FileSource.CONVERSATION_HISTORY]: 'Conversation',
      [FileSource.WORKFLOW_ARTIFACT]: 'Workflow',
      [FileSource.AGENT_GENERATED]: 'Agent Generated',
      [FileSource.SYSTEM_IMPORT]: 'System Import',
      [FileSource.WEB_SCRAPE]: 'Web Scrape'
    };
    return labels[source];
  }

  formatFileSize(bytes: number): string {
    return this.knowledgebaseService.formatFileSize(bytes);
  }

  // Drag and drop
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer?.files.length) {
      Array.from(event.dataTransfer.files).forEach(file => this.uploadFile(file));
    }
  }

  // Semantic search
  performSemanticSearch(query: string): void {
    if (!query.trim()) return;
    
    this.isLoading = true;
    this.knowledgebaseService.semanticSearch(query).subscribe({
      next: (results) => {
        // Results will be shown in the UI
        this.isLoading = false;
      },
      error: () => {
        this.snackBar.open('Semantic search failed', 'Close', {
          duration: 5000
        });
        this.isLoading = false;
      }
    });
  }

  // Semantic search dialog
  openSemanticSearch(): void {
    this.dialog.open(this.semanticSearchDialog, {
      panelClass: 'glass-effect',
      width: '400px'
    });
  }

  // Helper to map index -> TabView for template clean parsing
  tabIndexToView(idx: number): TabView {
    return (['personal', 'global', 'activity'][idx] ?? 'personal') as TabView;
  }

  // Helper to safely fetch source color
  getSourceColor(source: FileSource): string {
    return this.sourceColors[source] ?? '#ffffff';
  }

  getSourceBgColor(source: FileSource): string {
    return `${this.getSourceColor(source)}20`;
  }
}
