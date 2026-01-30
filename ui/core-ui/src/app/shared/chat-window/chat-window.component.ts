import { Component, ViewChild, ElementRef, Input, Output, EventEmitter, OnChanges, SimpleChanges, NgZone } from '@angular/core';
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
import { trigger, state, style, transition, animate } from '@angular/animations';

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
  encapsulation: ViewEncapsulation.None,
  animations: [
    trigger('slideDown', [
      transition(':enter', [
        style({ height: 0, opacity: 0, overflow: 'hidden' }),
        animate('200ms ease-out', style({ height: '*', opacity: 1 }))
      ]),
      transition(':leave', [
        style({ height: '*', opacity: 1, overflow: 'hidden' }),
        animate('200ms ease-in', style({ height: 0, opacity: 0 }))
      ])
    ])
  ]
})
export class ChatWindowComponent implements OnChanges {
  @Input() conversationId?: string;
  @Input() showTitle = true; // Show title in header (true for dashboard, false for conversations page)
  @Output() conversationIdChange = new EventEmitter<string>();

  conversationTitle = ''; // Generated conversation title
  isTitleGenerating = false; // Loading state for title generation
  messages: {
    sender: 'user' | 'assistant';
    text: string;
    thinking?: string; // AI reasoning process
    isStatus?: boolean;
    thinkingExpanded?: boolean; // UI state for thinking section
  }[] = [];
  newMessage = '';
  isStreaming = false;
  statusMessage = ''; // For displaying loading/heartbeat status
  defaultThinkingExpanded = false; // Global preference for showing thinking

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

  // RAG embeddings provider/model (independent of chat provider)
  ragEmbeddingProvider: 'openai' | 'local' = 'openai';
  ragLocalModels: string[] = ['nomic-embed-text', 'mxbai-embed-large', 'embedding-gemma'];
  ragSelectedLocalModel: string = this.ragLocalModels[0];

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
    private readonly kbService: KnowledgebaseService,
    private readonly zone: NgZone
  ) {
    // Load knowledgebase files for selection
    this.kbService.files$.subscribe(files => {
      this.kbFiles = files;
    });
    // Kick off initial load
    this.kbService.loadFiles().subscribe();

    // Load thinking preference from localStorage
    const savedPref = localStorage.getItem('core-thinking-expanded');
    if (savedPref !== null) {
      this.defaultThinkingExpanded = savedPref === 'true';
    }
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
        this.messages.push({ sender: 'assistant', text: 'Local LLM service is not reachable. Start Docker or select OpenAI.', thinking: '', thinkingExpanded: false });
      },
    });
  }

  pullLocalModel(): void {
    if (this.selectedProvider !== 'ollama' || !this.selectedModel || this.isPullingModel) return;
    this.isPullingModel = true;
    const name = this.selectedModel;
    this.messages.push({ sender: 'assistant', text: `Pulling local model: ${name} …`, thinking: '', thinkingExpanded: false });
    this.http.post<{ status: string }>(`${this._apiUrl}/local-llm/pull`, { name }).subscribe({
      next: () => {
        this.messages.push({ sender: 'assistant', text: `Model ready: ${name}`, thinking: '', thinkingExpanded: false });
        this.isPullingModel = false;
        this.loadLocalModels();
      },
      error: (err) => {
        this.messages.push({ sender: 'assistant', text: `Model pull failed: ${err?.error || err}`, thinking: '', thinkingExpanded: false });
        this.isPullingModel = false;
      }
    });
  }

  pullArbitraryLocalModel(): void {
    if (this.selectedProvider !== 'ollama' || this.isPullingModel) return;
    const name = (this.newLocalModelName || '').trim();
    if (!name) return;
    this.isPullingModel = true;
    this.messages.push({ sender: 'assistant', text: `Pulling local model: ${name} …`, thinking: '', thinkingExpanded: false });
    this.http.post<{ status: string; already_present?: boolean }>(`${this._apiUrl}/local-llm/pull`, { name }).subscribe({
      next: (res) => {
        const suffix = res?.already_present ? ' (already present)' : '';
        this.messages.push({ sender: 'assistant', text: `Model ready: ${name}${suffix}`, thinking: '', thinkingExpanded: false });
        // Ensure it appears in the dropdown and select it
        if (!this.models.includes(name)) {
          this.models = [name, ...this.models];
        }
        this.selectedModel = name;
        this.isPullingModel = false;
        this.newLocalModelName = '';
      },
      error: (err) => {
        this.messages.push({ sender: 'assistant', text: `Model pull failed: ${err?.error || err}`, thinking: '', thinkingExpanded: false });
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
          .get<{ id: string; title?: string; messages: { role: string; content: string; thinking?: string }[] }>(`${this._apiUrl}/conversations/${this.conversationId}`)
          .subscribe((data) => {
            const mapped = data.messages.map((m) => ({
              sender: m.role as 'user' | 'assistant',
              text: m.content,
              thinking: m.thinking || '',
              thinkingExpanded: this.defaultThinkingExpanded,
            }));

            // Update conversation title
            this.conversationTitle = data.title || '';

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
      this.messages.push({ sender: 'user', text: content, thinking: '', thinkingExpanded: false });
      this.newMessage = '';

      const assistantIdx = this.messages.push({
        sender: 'assistant',
        text: '',
        thinking: '',
        thinkingExpanded: this.defaultThinkingExpanded
      }) - 1;

      this.isStreaming = true;
      this.chatService
        .sendMessage(content, this.selectedModel, convId ?? this.conversationId, {
          kbMode: this.kbMode,
          kbFileId: this.kbMode === 'file' ? this.kbFileId : undefined,
          provider: this.selectedProvider,
          kbEmbeddingProvider: this.kbMode !== 'none' ? this.ragEmbeddingProvider : undefined,
          kbLocalModel: this.kbMode !== 'none' && this.ragEmbeddingProvider === 'local' ? this.ragSelectedLocalModel : undefined,
        })
        .subscribe({
          next: (jsonStr) => {
            this.zone.run(() => {
              try {
                const event = JSON.parse(jsonStr);
                const eventType = event.type || 'message';
                const eventData = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;

                // Handle conversation ID metadata
                if (eventData?.meta?.conversation_id) {
                  this.conversationId = eventData.meta.conversation_id;
                  this.conversationIdChange.emit(this.conversationId);
                  return;
                }

                // Handle different event types
                switch (eventType) {
                  case 'message':
                    // Regular message delta (actual response content)
                    const token = eventData?.delta ?? '';
                    this.messages[assistantIdx].text += token;
                    this.statusMessage = ''; // Clear status once we get content
                    break;

                  case 'thinking':
                    // AI reasoning process delta
                    const thinkingToken = eventData?.delta ?? '';
                    if (!this.messages[assistantIdx].thinking) {
                      this.messages[assistantIdx].thinking = '';
                    }
                    this.messages[assistantIdx].thinking! += thinkingToken;
                    break;

                  case 'status':
                    // Status update (e.g., "Connecting to model...")
                    this.statusMessage = eventData?.message || 'Processing...';
                    break;

                  case 'heartbeat':
                    // Periodic heartbeat during slow operations
                    this.statusMessage = eventData?.message || `Generating response... (${eventData?.elapsed || 0}s)`;
                    break;

                  case 'warning':
                    // Non-fatal warning (e.g., KB retrieval failed)
                    console.warn('Chat warning:', eventData?.message);
                    // Could show a toast notification here
                    break;

                  case 'error':
                    // Error event
                    this.messages[assistantIdx].text = `[error] ${eventData?.message || eventData?.error || 'Unknown error'}`;
                    this.statusMessage = '';
                    break;
                }

                this.scrollToBottom();
              } catch (err) {
                console.error('Failed to parse SSE event:', err, jsonStr);
              }
            });
          },
          error: (err) => {
            this.zone.run(() => {
              this.messages[assistantIdx].text = `[error] ${err}`;
              this.isStreaming = false;
              this.statusMessage = '';
              this.scrollToBottom();
            });
          },
          complete: () => {
            this.zone.run(() => {
              this.isStreaming = false;
              this.statusMessage = '';
              this.scrollToBottom();

              // After first response completes, poll for title
              if (this.messages.filter(m => m.sender === 'assistant').length === 1) {
                this.isTitleGenerating = true;
                this.pollForTitle();
              }
            });
          },
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

  toggleThinking(messageIndex: number): void {
    const msg = this.messages[messageIndex];
    if (msg) {
      msg.thinkingExpanded = !msg.thinkingExpanded;
    }
  }

  toggleGlobalThinkingPreference(): void {
    this.defaultThinkingExpanded = !this.defaultThinkingExpanded;
    localStorage.setItem('core-thinking-expanded', String(this.defaultThinkingExpanded));

    // Apply to all existing messages
    this.messages.forEach(msg => {
      if (msg.sender === 'assistant' && msg.thinking) {
        msg.thinkingExpanded = this.defaultThinkingExpanded;
      }
    });
  }

  private pollForTitle(): void {
    if (!this.conversationId) return;

    // Poll for title every 2 seconds for up to 30 seconds
    let attempts = 0;
    const maxAttempts = 15;

    const interval = setInterval(() => {
      attempts++;

      this.http
        .get<{ id: string; title?: string }>(`${this._apiUrl}/conversations/${this.conversationId}`)
        .subscribe({
          next: (data) => {
            if (data.title && data.title !== 'New Conversation') {
              this.conversationTitle = data.title;
              this.isTitleGenerating = false;
              clearInterval(interval);
            } else if (attempts >= maxAttempts) {
              this.isTitleGenerating = false;
              clearInterval(interval);
            }
          },
          error: () => {
            this.isTitleGenerating = false;
            clearInterval(interval);
          }
        });
    }, 2000);
  }

  private scrollToBottom(): void {
    queueMicrotask(() => {
      if (this.scrollContainer) {
        this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
      }
    });
  }
}
