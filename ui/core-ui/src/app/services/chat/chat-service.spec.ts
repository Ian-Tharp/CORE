import { TestBed } from '@angular/core/testing';

import { ChatService } from './chat-service';
import { AppConfigService } from '../config/app-config.service';

describe('ChatService', () => {
  let service: ChatService;
  let fetchMock: jest.Mock;

  // Deterministic stream URL so assertions do not depend on the real config.
  const chatStreamUrl = 'http://localhost:8000/chat/stream';

  // Minimal AppConfigService stand-in exposing only what ChatService reads.
  const configStub: Pick<AppConfigService, 'chatStreamUrl'> = {
    chatStreamUrl
  };

  beforeEach(() => {
    // ChatService streams over the native fetch API (not HttpClient), so we
    // stub fetch to immediately complete the stream with an empty body.
    fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: { get: () => null },
      body: {
        getReader: () => ({
          read: () => Promise.resolve({ value: undefined, done: true })
        })
      }
    } as unknown as Response);
    global.fetch = fetchMock as unknown as typeof fetch;

    TestBed.configureTestingModule({
      providers: [
        ChatService,
        { provide: AppConfigService, useValue: configStub }
      ]
    });

    service = TestBed.inject(ChatService);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should format payload correctly and make POST request', () => {
    const model = 'gpt-4o';
    const content = 'Hello world';

    service.sendMessage(content, model).subscribe();

    // The service issues exactly one fetch to the configured stream URL.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];

    expect(url).toBe(chatStreamUrl);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({
      model,
      messages: [
        {
          role: 'user',
          content
        }
      ],
      stream: true
    });
  });
});
