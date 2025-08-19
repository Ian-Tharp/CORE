import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { Observable, BehaviorSubject, Subject, of, throwError } from 'rxjs';
import { map, tap, catchError, finalize, filter } from 'rxjs/operators';
import {
  KnowledgeFile,
  VectorEmbedding,
  EmbeddingStats,
  FileTag,
  FileMetadata,
  KnowledgebaseFilter,
  KnowledgebaseStats,
  FileUploadRequest,
  FileStatus,
  FileSource,
  ActivityLog
} from '../../models/knowledgebase.models';

@Injectable({
  providedIn: 'root'
})
export class KnowledgebaseService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8001/knowledgebase';

  // State management
  private filesSubject = new BehaviorSubject<KnowledgeFile[]>([]);
  private statsSubject = new BehaviorSubject<KnowledgebaseStats | null>(null);
  private uploadProgressSubject = new Subject<{ fileId: string; progress: number }>();
  private uploadStageSubject = new Subject<{ fileId: string; stage: string }>();
  private activeFiltersSubject = new BehaviorSubject<KnowledgebaseFilter>({});

  // Public observables
  files$ = this.filesSubject.asObservable();
  stats$ = this.statsSubject.asObservable();
  uploadProgress$ = this.uploadProgressSubject.asObservable();
  uploadStage$ = this.uploadStageSubject.asObservable();
  activeFilters$ = this.activeFiltersSubject.asObservable();

  // Available tags
  private availableTagsSubject = new BehaviorSubject<FileTag[]>([]);
  availableTags$ = this.availableTagsSubject.asObservable();

  constructor() {
    // Defer initialization to avoid potential circular dependency issues
    setTimeout(() => {
      this.initializeService();
    }, 0);
  }

  private initializeService(): void {
    // Only load if we have a valid HTTP client
    if (this.http) {
      this.loadFiles().subscribe();
      this.loadStats().subscribe();
      this.loadAvailableTags().subscribe();
    }
  }

  // File operations
  loadFiles(filter?: KnowledgebaseFilter): Observable<KnowledgeFile[]> {
    const params = this.buildFilterParams(filter || this.activeFiltersSubject.value);
    
    return this.http.get<KnowledgeFile[]>(`${this.apiUrl}/files`, { params }).pipe(
      map(files => files.map(file => ({
        ...file,
        uploadDate: new Date(file.uploadDate),
        lastModified: new Date(file.lastModified)
      }))),
      tap(files => this.filesSubject.next(files)),
      catchError(this.handleError)
    );
  }

  getFile(fileId: string): Observable<KnowledgeFile> {
    return this.http.get<KnowledgeFile>(`${this.apiUrl}/files/${fileId}`).pipe(
      map(file => ({
        ...file,
        uploadDate: new Date(file.uploadDate),
        lastModified: new Date(file.lastModified)
      })),
      catchError(this.handleError)
    );
  }

  uploadFile(request: FileUploadRequest): Observable<KnowledgeFile> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('data', JSON.stringify({
      tags: request.tags,
      description: request.description,
      isGlobal: request.isGlobal,
      metadata: request.metadata,
      processImmediately: request.processImmediately
    }));

    // Emit initial stage
    this.uploadStageSubject.next({ fileId: request.file.name, stage: 'uploading' });

    return this.http.post<KnowledgeFile>(`${this.apiUrl}/upload`, formData, {
      reportProgress: true,
      observe: 'events'
    }).pipe(
      map((event: any) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          const progress = Math.round(100 * event.loaded / event.total);
          this.uploadProgressSubject.next({ fileId: request.file.name, progress });
          this.uploadStageSubject.next({ fileId: request.file.name, stage: 'uploading' });
          return null;
        }
        if (event.type === HttpEventType.Response) {
          this.uploadProgressSubject.next({ fileId: request.file.name, progress: 100 });
          this.uploadStageSubject.next({ fileId: request.file.name, stage: request.processImmediately ? 'processing' : 'finalizing' });
          return event.body as KnowledgeFile;
        }
        return null;
      }),
      filter((file): file is KnowledgeFile => file !== null),
      tap(file => {
        if (file) {
          const currentFiles = this.filesSubject.value;
          this.filesSubject.next([file, ...currentFiles]);
          this.loadStats();
          this.uploadStageSubject.next({ fileId: request.file.name, stage: 'complete' });
          // After upload, if processing or missing model/chunks, refresh until ready
          this.refreshFileUntilReady(file.id).subscribe();
        }
      }),
      catchError((err) => {
        this.uploadStageSubject.next({ fileId: request.file.name, stage: 'error' });
        return this.handleError(err);
      })
    );
  }

  deleteFile(fileId: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/files/${fileId}`).pipe(
      tap(() => {
        const currentFiles = this.filesSubject.value;
        this.filesSubject.next(currentFiles.filter(f => f.id !== fileId));
        this.loadStats();
      }),
      catchError(this.handleError)
    );
  }

  // Vector embedding operations
  getEmbeddingStats(fileId: string): Observable<EmbeddingStats> {
    return this.http.get<EmbeddingStats>(`${this.apiUrl}/files/${fileId}/embeddings/stats`).pipe(
      catchError(this.handleError)
    );
  }

  getEmbeddings(fileId: string): Observable<VectorEmbedding[]> {
    return this.http.get<VectorEmbedding[]>(`${this.apiUrl}/files/${fileId}/embeddings`).pipe(
      catchError(this.handleError)
    );
  }

  processFileEmbeddings(fileId: string): Observable<EmbeddingStats> {
    // Optimistically mark processing, then flip to READY on success
    this.updateFileStatus(fileId, FileStatus.PROCESSING);
    return this.http.post<any>(`${this.apiUrl}/files/${fileId}/process`, {}).pipe(
      tap((resp) => {
        // Backend returns { fileId, status: 'ready' }
        if (resp && resp.status && String(resp.status).toLowerCase() === 'ready') {
          this.updateFileStatus(fileId, FileStatus.READY);
        }
        // Ensure model/chunk counts are refreshed
        this.refreshFileUntilReady(fileId).subscribe();
      }),
      catchError(this.handleError)
    );
  }

  // Re-extract a better title for the document
  reextractTitle(fileId: string): Observable<{ fileId: string; title?: string; updated: boolean }> {
    return this.http.post<{ fileId: string; title?: string; updated: boolean }>(`${this.apiUrl}/files/${fileId}/reextract-title`, {}).pipe(
      tap(res => {
        if (res.updated && res.title) {
          const current = this.filesSubject.value.map(f => f.id === fileId ? { ...f, title: res.title } : f);
          this.filesSubject.next(current);
        }
      }),
      catchError(this.handleError)
    );
  }

  // Metadata and tagging
  getFileMetadata(fileId: string): Observable<FileMetadata> {
    return this.http.get<FileMetadata>(`${this.apiUrl}/files/${fileId}/metadata`).pipe(
      catchError(this.handleError)
    );
  }

  updateFileMetadata(fileId: string, metadata: Partial<FileMetadata>): Observable<FileMetadata> {
    return this.http.patch<FileMetadata>(`${this.apiUrl}/files/${fileId}/metadata`, metadata).pipe(
      catchError(this.handleError)
    );
  }

  addTag(fileId: string, tag: string): Observable<void> {
    return this.http.post<void>(`${this.apiUrl}/files/${fileId}/tags`, { tag }).pipe(
      catchError(this.handleError)
    );
  }

  removeTag(fileId: string, tagId: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/files/${fileId}/tags/${tagId}`).pipe(
      catchError(this.handleError)
    );
  }

  // Search and filtering
  searchFiles(query: string): Observable<KnowledgeFile[]> {
    return this.http.get<KnowledgeFile[]>(`${this.apiUrl}/search`, {
      params: { q: query }
    }).pipe(
      map(files => files.map(file => ({
        ...file,
        uploadDate: new Date(file.uploadDate),
        lastModified: new Date(file.lastModified)
      }))),
      catchError(this.handleError)
    );
  }

  semanticSearch(query: string, limit: number = 10): Observable<Array<KnowledgeFile & { similarity: number }>> {
    return this.http.post<Array<KnowledgeFile & { similarity: number }>>(`${this.apiUrl}/semantic-search`, {
      query,
      limit
    }).pipe(
      catchError(this.handleError)
    );
  }

  applyFilter(filter: KnowledgebaseFilter): void {
    this.activeFiltersSubject.next(filter);
    this.loadFiles(filter);
  }

  clearFilters(): void {
    this.activeFiltersSubject.next({});
    this.loadFiles();
  }

  // Stats and activity
  loadStats(): Observable<KnowledgebaseStats> {
    return this.http.get<KnowledgebaseStats>(`${this.apiUrl}/stats`).pipe(
      tap(stats => this.statsSubject.next(stats)),
      catchError((error) => {
        console.error('Failed to load stats:', error);
        // Return default stats when API is not available
        const defaultStats: KnowledgebaseStats = {
          totalFiles: 0,
          totalSize: 0,
          filesByType: {},
          filesBySource: {} as Record<FileSource, number>,
          totalEmbeddings: 0,
          processingQueue: 0,
          recentActivity: []
        };
        this.statsSubject.next(defaultStats);
        return of(defaultStats);
      })
    );
  }

  getRecentActivity(limit: number = 20): Observable<ActivityLog[]> {
    return this.http.get<ActivityLog[]>(`${this.apiUrl}/activity`, {
      params: { limit: limit.toString() }
    }).pipe(
      map(activities => activities.map(activity => ({
        ...activity,
        timestamp: new Date(activity.timestamp)
      }))),
      catchError(this.handleError)
    );
  }

  // Tag management
  loadAvailableTags(): Observable<FileTag[]> {
    return this.http.get<FileTag[]>(`${this.apiUrl}/tags`).pipe(
      tap(tags => this.availableTagsSubject.next(tags)),
      catchError(this.handleError)
    );
  }

  createTag(tag: Omit<FileTag, 'id'>): Observable<FileTag> {
    return this.http.post<FileTag>(`${this.apiUrl}/tags`, tag).pipe(
      tap(newTag => {
        const currentTags = this.availableTagsSubject.value;
        this.availableTagsSubject.next([...currentTags, newTag]);
      }),
      catchError(this.handleError)
    );
  }

  // Helper methods
  private buildFilterParams(filter: KnowledgebaseFilter): any {
    const params: any = {};
    
    if (filter.searchQuery) params.q = filter.searchQuery;
    if (filter.tags?.length) params.tags = filter.tags.join(',');
    if (filter.fileTypes?.length) params.types = filter.fileTypes.join(',');
    if (filter.sources?.length) params.sources = filter.sources.join(',');
    if (filter.status?.length) params.status = filter.status.join(',');
    if (filter.isGlobal !== undefined) params.global = filter.isGlobal.toString();
    if (filter.userId) params.userId = filter.userId;
    if (filter.dateRange) {
      params.startDate = filter.dateRange.start.toISOString();
      params.endDate = filter.dateRange.end.toISOString();
    }
    
    return params;
  }

  private updateFileStatus(fileId: string, status: FileStatus): void {
    const currentFiles = this.filesSubject.value;
    const updatedFiles = currentFiles.map(file =>
      file.id === fileId ? { ...file, status } : file
    );
    this.filesSubject.next(updatedFiles);
  }

  // Poll the backend for file details until we see READY and metadata populated
  private refreshFileUntilReady(fileId: string, maxAttempts: number = 80, intervalMs: number = 1500): Observable<KnowledgeFile | null> {
    return new Observable<KnowledgeFile | null>((subscriber) => {
      let attempts = 0;
      const tick = () => {
        attempts += 1;
        this.getFile(fileId).subscribe({
          next: (file) => {
            // Update the file in the list
            const current = this.filesSubject.value.map(f => f.id === file.id ? {
              ...f,
              ...file
            } : f);
            this.filesSubject.next(current);
            const ready = String(file.status).toLowerCase() === 'ready';
            const metaPresent = !!(file.embeddingModel) && (file.chunkCount !== undefined);
            if ((ready && metaPresent) || attempts >= maxAttempts) {
              subscriber.next(file);
              subscriber.complete();
            } else {
              setTimeout(tick, intervalMs);
            }
          },
          error: () => {
            if (attempts >= maxAttempts) {
              subscriber.next(null);
              subscriber.complete();
            } else {
              setTimeout(tick, intervalMs);
            }
          }
        });
      };
      tick();
    });
  }

  private handleError(error: any): Observable<any> {
    console.error('Knowledgebase service error:', error);
    // Instead of throwing error, return empty data for development
    if (error.status === 404 || error.status === 0) {
      console.warn('API endpoint not available, returning empty data');
      return of([]);
    }
    if (error.status === 409) {
      return throwError(() => ({ ...error, message: 'duplicate' }));
    }
    return throwError(() => error);
  }

  // Utility methods for file size formatting
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
} 