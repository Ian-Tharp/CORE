import { Component, ViewChild, ElementRef, Input, Output, EventEmitter, OnChanges, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat/chat-service';
import { KnowledgebaseService } from '../../services/knowledgebase/knowledgebase.service';
import { KnowledgeFile } from '../../models/knowledgebase.models';
import { MarkdownModule } from 'ngx-markdown';
import { HttpClient } from '@angular/common/http';
import { ViewEncapsulation } from '@angular/core';
import { CdkTextareaAutosize, TextFieldModule } from '@angular/cdk/text-field';

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
    MarkdownModule,
    TextFieldModule
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

  // Provider + model selection
  readonly providers: Array<'openai' | 'ollama'> = ['openai', 'ollama'];
  selectedProvider: 'openai' | 'ollama' = 'openai';
  models: string[] = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4o', 'o3', 'o4-mini'];
  selectedModel = this.models[0];
  isPullingModel = false;
  newLocalModelName = '';

  // Knowledgebase RAG options
  kbMode: 'none' | 'all' | 'file' = 'none';
  kbFiles: KnowledgeFile[] = [];
  kbFileId?: string;

  // Reference to the scrolling container so we can auto-scroll.
  @ViewChild('scrollContainer', { static: false })
  private scrollContainer?: ElementRef<HTMLDivElement>;

  // Reference to the textarea autosize directive
  @ViewChild(CdkTextareaAutosize, { static: false })
  private autosize?: CdkTextareaAutosize;

  private readonly _apiUrl = 'http://localhost:8001';

  constructor(
    private readonly chatService: ChatService,
    private readonly http: HttpClient,
    private readonly kbService: KnowledgebaseService
  ) {
    // Load knowledgebase files for selection
    this.kbService.files$.subscribe(files => {
      this.kbFiles = files;
    });
    // Kick off initial load
    this.kbService.loadFiles().subscribe();
  }

  onProviderChange(): void {
    if (this.selectedProvider === 'ollama') {
      this.checkLocalHealth();
      this.loadLocalModels();
    } else if (this.selectedProvider === 'openai') {
      this.models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4o', 'o3', 'o4-mini'];
      this.selectedModel = this.models[0];
    }
  }

  private loadLocalModels(): void {
    this.http.get<{ models: Array<{ name: string }> }>(`${this._apiUrl}/local-llm/models`).subscribe({
      next: (res) => {
        this.models = (res.models || []).map((m) => m.name);
        if (this.models.length === 0) {
          this.models = ['gpt-oss:20b'];
        }
        this.selectedModel = this.models[0];
      },
      error: () => {
        // Fallback sensible default; user can still type
        this.models = ['gpt-oss:20b'];
        this.selectedModel = this.models[0];
      },
    });
  }

  private checkLocalHealth(): void {
    this.http.get<{ status: string }>(`${this._apiUrl}/local-llm/health`).subscribe({
      next: () => {
        // no-op when healthy
      },
      error: () => {
        // Surface a lightweight inline message for visibility
        this.messages.push({ sender: 'assistant', text: 'Local LLM service is not reachable. Start Docker or select OpenAI.' });
      },
    });
  }

  pullLocalModel(): void {
    if (this.selectedProvider !== 'ollama' || !this.selectedModel || this.isPullingModel) return;
    this.isPullingModel = true;
    const name = this.selectedModel;
    this.messages.push({ sender: 'assistant', text: `Pulling local model: ${name} …` });
    this.http.post<{ status: string }>(`${this._apiUrl}/local-llm/pull`, { name }).subscribe({
      next: () => {
        this.messages.push({ sender: 'assistant', text: `Model ready: ${name}` });
        this.isPullingModel = false;
        this.loadLocalModels();
      },
      error: (err) => {
        this.messages.push({ sender: 'assistant', text: `Model pull failed: ${err?.error || err}` });
        this.isPullingModel = false;
      }
    });
  }

  pullArbitraryLocalModel(): void {
    if (this.selectedProvider !== 'ollama' || this.isPullingModel) return;
    const name = (this.newLocalModelName || '').trim();
    if (!name) return;
    this.isPullingModel = true;
    this.messages.push({ sender: 'assistant', text: `Pulling local model: ${name} …` });
    this.http.post<{ status: string; already_present?: boolean }>(`${this._apiUrl}/local-llm/pull`, { name }).subscribe({
      next: (res) => {
        const suffix = res?.already_present ? ' (already present)' : '';
        this.messages.push({ sender: 'assistant', text: `Model ready: ${name}${suffix}` });
        // Ensure it appears in the dropdown and select it
        if (!this.models.includes(name)) {
          this.models = [name, ...this.models];
        }
        this.selectedModel = name;
        this.isPullingModel = false;
        this.newLocalModelName = '';
      },
      error: (err) => {
        this.messages.push({ sender: 'assistant', text: `Model pull failed: ${err?.error || err}` });
        this.isPullingModel = false;
      }
    });
  }

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
    const run = (convId?: string) => {
      this.messages.push({ sender: 'user', text: content });
      this.newMessage = '';

      const assistantIdx = this.messages.push({ sender: 'assistant', text: '' }) - 1;

      this.chatService
        .sendMessage(content, this.selectedModel, convId ?? this.conversationId, {
          kbMode: this.kbMode,
          kbFileId: this.kbMode === 'file' ? this.kbFileId : undefined,
          provider: this.selectedProvider,
        })
        .subscribe({
          next: (jsonStr) => {
            try {
              const data = JSON.parse(jsonStr);

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
    };

    // If we don't yet have a conversation id, create one first to avoid race
    // conditions when users send multiple rapid messages.
    if (!this.conversationId) {
      this.http.post<{ id: string }>(`${this._apiUrl}/conversations/`, {}).subscribe({
        next: (res) => {
          this.conversationId = res.id;
          this.conversationIdChange.emit(this.conversationId);
          run(this.conversationId);
        },
        error: () => run(undefined), // fallback to legacy behavior
      });
    } else {
      run(this.conversationId);
    }
  }

  onEnterKey(event: Event): void {
    // Cast to KeyboardEvent for type safety
    const keyboardEvent = event as KeyboardEvent;
    // Send message on Enter, but allow Shift+Enter for new lines
    if (keyboardEvent.key === 'Enter' && !keyboardEvent.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  private scrollToBottom(): void {
    queueMicrotask(() => {
      if (this.scrollContainer) {
        this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
      }
    });
  }
}
