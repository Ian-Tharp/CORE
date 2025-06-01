import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { MatButtonModule } from '@angular/material/button';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { HttpClient } from '@angular/common/http';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { FormsModule } from '@angular/forms';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDividerModule } from '@angular/material/divider';

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
  ]
})
export class ConversationsPageComponent implements OnInit {
  conversations: ConversationSummary[] = [];
  selectedConversationId?: string;
  editingConversationId?: string;
  editingTitle: string = '';

  private readonly _apiUrl = 'http://localhost:8001';

  constructor(private readonly http: HttpClient) {}

  ngOnInit(): void {
    this.refreshList();
  }

  refreshList(): void {
    this.http
      .get<ConversationSummary[]>(`${this._apiUrl}/conversations/`)
      .subscribe((data) => (this.conversations = data));
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

    this.http
      .patch(`${this._apiUrl}/conversations/${conv.id}`, {
        title: newTitle,
      })
      .subscribe(() => {
        conv.title = newTitle;
        this.editingConversationId = undefined;
      });
  }
}
