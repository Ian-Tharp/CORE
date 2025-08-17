import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { MatButtonModule } from '@angular/material/button';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { HttpClient } from '@angular/common/http';
import { EnginePlaygroundComponent } from './engine-playground.component';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { FormsModule } from '@angular/forms';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDividerModule } from '@angular/material/divider';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { catchError } from 'rxjs/operators';
import { of } from 'rxjs';

interface ConversationSummary {
  id: string;
  title: string;
  messages: number;
}

@Component({
  selector: 'app-conversations-page',
  templateUrl: './conversations-page.component.html',
  styleUrl: './conversations-page.component.scss',
  imports: [
    CommonModule,
    FormsModule,
    ChatWindowComponent,
    MatButtonModule,
    MatListModule,
    MatIconModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatTooltipModule,
    MatDividerModule,
    MatProgressSpinnerModule,
    EnginePlaygroundComponent,
  ]
})
export class ConversationsPageComponent implements OnInit {
  conversations: ConversationSummary[] = [];
  selectedConversationId?: string;
  editingConversationId?: string;
  editingTitle: string = '';
  
  // Connection status tracking
  isLoading: boolean = false;
  isConnected: boolean = true;
  connectionError: string = '';

  private readonly _apiUrl = 'http://localhost:8001';

  constructor(private readonly http: HttpClient) {}

  ngOnInit(): void {
    this.refreshList();
  }

  refreshList(): void {
    this.isLoading = true;
    this.connectionError = '';
    
    this.http
      .get<ConversationSummary[]>(`${this._apiUrl}/conversations/`)
      .pipe(
        catchError((error) => {
          console.error('Failed to load conversations:', error);
          this.isConnected = false;
          this.connectionError = this.getErrorMessage(error);
          return of([]); // Return empty array on error
        })
      )
      .subscribe((data) => {
        this.conversations = data;
        this.isConnected = data.length > 0 || this.connectionError === '';
        this.isLoading = false;
        
        // If we successfully got data, mark as connected
        if (data.length > 0 || this.connectionError === '') {
          this.isConnected = true;
        }
      });
  }

  private getErrorMessage(error: any): string {
    if (error.status === 0) {
      return 'Unable to connect to the backend service. Please ensure the Python backend is running on localhost:8001.';
    } else if (error.status >= 500) {
      return 'Backend server error. Please check the server logs.';
    } else if (error.status >= 400) {
      return 'Request error. Please try again.';
    } else {
      return 'Connection failed. Please check your network connection and try again.';
    }
  }

  retryConnection(): void {
    this.refreshList();
  }

  selectConversation(id: string): void {
    this.selectedConversationId = id;
    this.editingConversationId = undefined;
  }

  startNewConversation(): void {
    // Simply clear selection; chat window will create new conversation on first send.
    this.selectedConversationId = undefined;
  }

  onConversationCreated(id: string): void {
    this.selectedConversationId = id;
    this.refreshList();
  }

  beginEdit(conv: ConversationSummary, event: Event): void {
    event.stopPropagation();
    this.editingConversationId = conv.id;
    this.editingTitle = conv.title;
  }

  saveTitle(conv: ConversationSummary): void {
    const newTitle = this.editingTitle.trim();
    if (!newTitle || newTitle === conv.title) {
      this.editingConversationId = undefined;
      return;
    }

    // Optimistically update UI; revert on error
    const previousTitle = conv.title;
    conv.title = newTitle;

    this.http
      .patch(`${this._apiUrl}/conversations/${conv.id}`, {
        title: newTitle,
      })
      .pipe(
        catchError((error) => {
          console.error('Failed to update conversation title:', error);
          // Revert title on error
          conv.title = previousTitle;
          this.editingConversationId = undefined;
          return of(null);
        })
      )
      .subscribe(() => {
        // 204 No Content returns null body; we already updated optimistically
        this.editingConversationId = undefined;
      });
  }
}
