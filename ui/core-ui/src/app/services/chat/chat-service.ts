import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

import { UserInput } from '../../models/UserInput';

export type ChatMessage = UserInput & {
  id: string;
  timestamp: Date;
};

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private readonly _messages$: BehaviorSubject<ChatMessage[]> = new BehaviorSubject<ChatMessage[]>([]);

  /**
   * Base URL for the backend FastAPI service. In a real-world application this would
   * come from an environment file but is hard-coded here for brevity.
   */
  private readonly _apiUrl = 'http://localhost:8001';

  constructor() {}

  get messages$(): Observable<ChatMessage[]> {
    return this._messages$.asObservable();
  }

  addMessage(input: UserInput): void {
    const newMessage: ChatMessage = {
      ...input,
      id: this._generateId(),
      timestamp: new Date(),
    };
    const updatedMessages = [...this._messages$.value, newMessage];
    this._messages$.next(updatedMessages);
  }

  clearMessages(): void {
    this._messages$.next([]);
  }

  private _generateId(): string {
    return (
      Date.now().toString(36) +
      Math.random().toString(36).substring(2, 10)
    );
  }

  /**
   * Sends the current chat history to the backend in the shape expected by the
   * FastAPI controller (`ChatRequest` in the Python code-base).
   *
   * The backend streams [text/event-stream] data; however, for unit-testing and
   * to keep this example simple we request the response as plain text. Consumers
   * can parse Server-Sent Event chunks downstream as needed.
   *
   * @param content The content of the user message to send.
   * @param model   The OpenAI model the backend should use (e.g. `gpt-4o`).
   */
  sendMessage(
    content: string,
    model: string,
    conversationId?: string,
    options?: { kbMode?: 'none' | 'all' | 'file'; kbFileId?: string; provider?: 'openai' | 'ollama' | 'local' }
  ): Observable<string> {
    // 1. Persist the user message locally so that it is part of the chat history.
    const userInput: UserInput = { role: 'user', content };
    this.addMessage(userInput);

    // 2. Build the request payload by trimming local bookkeeping fields.
    const payload: Record<string, unknown> = {
      model,
      messages: this._messages$.value.map(({ role, content: msgContent }) => ({
        role,
        content: msgContent,
      })),
      stream: true,
    };

    if (conversationId) {
      payload["conversation_id"] = conversationId;
    }

    if (options?.kbMode && options.kbMode !== 'none') {
      payload["kb_mode"] = options.kbMode;
      if (options.kbMode === 'file' && options.kbFileId) {
        payload["kb_file_id"] = options.kbFileId;
      }
    }

    // Route to the desired provider; default to OpenAI
    if (options?.provider) {
      const p = options.provider === 'local' ? 'ollama' : options.provider;
      payload["provider"] = p;
    }

    return new Observable<string>((observer) => {
      fetch(`${this._apiUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
        .then((response) => {
          if (!response.ok || !response.body) {
            observer.error(
              `HTTP error ${response.status}: ${response.statusText}`,
            );
            return;
          }

          // If the caller did not yet have a conversation id, attempt to read
          // it from the response headers.
          const convId = response.headers.get('X-Conversation-Id');
          if (!conversationId && convId) {
            // Emit a synthetic SSE chunk so that consumers can capture the id.
            observer.next(JSON.stringify({ meta: { conversation_id: convId } }));
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          const readChunk = () => {
            reader.read().then(({ value, done }) => {
              if (done) {
                observer.complete();
                return;
              }

              const text = decoder.decode(value, { stream: true });

              // SSE chunks are separated by double newlines
              for (const part of text.split('\n\n')) {
                const trimmed = part.trim();
                if (!trimmed.startsWith('data:')) continue;

                const jsonStr = trimmed.replace(/^data:\s*/, '');
                if (jsonStr === '[DONE]') continue;

                observer.next(jsonStr);
              }

              readChunk();
            });
          };

          readChunk();
        })
        .catch((err) => observer.error(err));
    });
  }
}

