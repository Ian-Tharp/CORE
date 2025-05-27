import { Component, ViewChild, ElementRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat/chat-service';
import { MarkdownModule } from 'ngx-markdown';

@Component({
  selector: 'app-chat-window',
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
  styleUrl: './chat-window.component.scss'
})
export class ChatWindowComponent {
  messages: { sender: 'user' | 'assistant'; text: string }[] = [];
  newMessage = '';

  // Available OpenAI models – keep in sync with the backend.
  readonly models: string[] = ['gpt-4o', 'gpt-4o-mini', 'o1-preview-2024-09-12', 'gpt-4.1'];
  selectedModel = this.models[0];

  // Reference to the scrolling container so we can auto-scroll.
  @ViewChild('scrollContainer', { static: false })
  private scrollContainer?: ElementRef<HTMLDivElement>;

  constructor(private readonly chatService: ChatService) {}

  sendMessage(): void {
    const content = this.newMessage.trim();
    if (!content) return;
    this.messages.push({ sender: 'user', text: content });
    this.newMessage = '';

    // Placeholder assistant message that will be updated as chunks arrive.
    const assistantIdx = this.messages.push({ sender: 'assistant', text: '' }) - 1;

    this.chatService.sendMessage(content, this.selectedModel).subscribe({
      next: (jsonStr) => {
        try {
          const data = JSON.parse(jsonStr);
          const token = data?.choices?.[0]?.delta ?? '';
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
