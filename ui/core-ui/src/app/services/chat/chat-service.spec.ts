import { TestBed } from '@angular/core/testing';
import {
  HttpClientTestingModule,
  HttpTestingController,
} from '@angular/common/http/testing';

import { ChatService } from './chat-service';

describe('ChatService', () => {
  let service: ChatService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
    });

    service = TestBed.inject(ChatService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should format payload correctly and make POST request', () => {
    const model = 'gpt-4o';
    const content = 'Hello world';

    service.sendMessage(content, model).subscribe();

    const req = httpMock.expectOne('http://localhost:8000/chat/stream');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      model,
      messages: [
        {
          role: 'user',
          content,
        },
      ],
      stream: true,
    });

    // Respond with an empty string to complete the observable.
    req.flush('');
  });
});
