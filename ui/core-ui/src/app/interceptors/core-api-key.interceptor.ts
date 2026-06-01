import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';

import { AppConfigService } from '../services/config/app-config.service';

export const coreApiKeyInterceptor: HttpInterceptorFn = (request, next) => {
  const config = inject(AppConfigService);

  if (!config.isCoreApiUrl(request.url) || request.headers.has('X-API-Key')) {
    return next(request);
  }

  return next(request.clone({
    setHeaders: {
      'X-API-Key': config.apiKey
    }
  }));
};
