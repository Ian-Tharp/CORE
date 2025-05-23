import { Component, ViewChild, ElementRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chat-window',
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule
  ],
  templateUrl: './chat-window.component.html',
  styleUrl: './chat-window.component.scss'
})
export class ChatWindowComponent {
  messages: { sender: 'user' | 'assistant'; text: string }[] = [];
  newMessage = '';

  @ViewChild('scrollContainer') private scrollContainer?: ElementRef<HTMLDivElement>;

  sendMessage(): void {
    const content = this.newMessage.trim();
    if (!content) return;
    this.messages.push({ sender: 'user', text: content });
    this.newMessage = '';

    // Simple simulated assistant response
    setTimeout(() => {
      this.messages.push({ sender: 'assistant', text: 'Acknowledged: ' + content });
      this.scrollToBottom();
    }, 400);

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
