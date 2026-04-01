import { TestBed } from '@angular/core/testing';
import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { coreApiKeyInterceptor } from './core-api-key.interceptor';

describe('coreApiKeyInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([coreApiKeyInterceptor])),
        provideHttpClientTesting(),
      ],
    });

    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should attach the CORE API key to backend requests', () => {
    // Arrange / Act
    http.get('http://localhost:8001/knowledgebase/files').subscribe();

    // Assert
    const request = httpMock.expectOne('http://localhost:8001/knowledgebase/files');
    expect(request.request.headers.get('X-API-Key')).toBe('core_dev_key');
    request.flush([]);
  });

  it('should leave non-CORE requests unchanged', () => {
    // Arrange / Act
    http.get('https://example.com/health').subscribe();

    // Assert
    const request = httpMock.expectOne('https://example.com/health');
    expect(request.request.headers.has('X-API-Key')).toBe(false);
    request.flush({});
  });
});
