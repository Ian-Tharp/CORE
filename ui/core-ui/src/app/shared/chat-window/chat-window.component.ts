import { Component, ViewChild, ElementRef, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat/chat-service';
import { MarkdownModule } from 'ngx-markdown';
import { HttpClient } from '@angular/common/http';
import { ViewEncapsulation } from '@angular/core';

@Component({
  selector: 'app-chat-window',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MarkdownModule
  ],
  templateUrl: './chat-window.component.html',
  styleUrl: './chat-window.component.scss',
  encapsulation: ViewEncapsulation.None
})
export class ChatWindowComponent implements OnChanges {
  @Input() conversationId?: string;
  @Output() conversationIdChange = new EventEmitter<string>();

  messages: { sender: 'user' | 'assistant'; text: string }[] = [];
  newMessage = '';

  // Available OpenAI models – keep in sync with the backend.
  readonly models: string[] = ['gpt-4o', 'o3', 'o4-mini', 'gpt-4.1'];
  selectedModel = this.models[0];

  // Reference to the scrolling container so we can auto-scroll.
  @ViewChild('scrollContainer', { static: false })
  private scrollContainer?: ElementRef<HTMLDivElement>;

  private readonly _apiUrl = 'http://localhost:8001';

  constructor(private readonly chatService: ChatService, private readonly http: HttpClient) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['conversationId'] && !changes['conversationId'].firstChange) {
      // When the conversation id changes, reset local message list
      this.messages = [];
      this.chatService.clearMessages();

      // If a valid conversation id is provided, load its history.
      if (this.conversationId) {
        this.http
          .get<{ id: string; messages: { role: string; content: string }[] }>(`${this._apiUrl}/conversations/${this.conversationId}`)
          .subscribe((data) => {
            const mapped = data.messages.map((m) => ({
              sender: m.role as 'user' | 'assistant',
              text: m.content,
            }));

            // Update local display
            this.messages = mapped;

            // Sync with ChatService history so that subsequent API calls have
            // the full context.
            this.chatService.clearMessages();
            mapped.forEach((msg) =>
              this.chatService.addMessage({
                role: msg.sender,
                content: msg.text,
              })
            );

            this.scrollToBottom();
          });
      }
    }
  }

  sendMessage(): void {
    const content = this.newMessage.trim();
    if (!content) return;
    this.messages.push({ sender: 'user', text: content });
    this.newMessage = '';

    // Placeholder assistant message that will be updated as chunks arrive.
    const assistantIdx = this.messages.push({ sender: 'assistant', text: '' }) - 1;

    this.chatService
      .sendMessage(content, this.selectedModel, this.conversationId)
      .subscribe({
        next: (jsonStr) => {
          try {
            const data = JSON.parse(jsonStr);

            // Handle synthetic meta message for newly created conversations
            if (data?.meta?.conversation_id) {
              this.conversationId = data.meta.conversation_id;
              this.conversationIdChange.emit(this.conversationId);
              return;
            }

            const token = data?.delta ?? '';
            this.messages[assistantIdx].text += token;
          } catch {
            /* ignore malformed chunks */
          }
        },
        error: (err) => {
          this.messages[assistantIdx].text = `[error] ${err}`;
          this.scrollToBottom();
        },
        complete: () => this.scrollToBottom(),
      });

    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    queueMicrotask(() => {
      if (this.scrollContainer) {
        this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
      }
    });
  }
}
