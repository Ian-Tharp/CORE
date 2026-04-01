import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { of } from 'rxjs';

import { AppComponent } from './app.component';
import { DiscordBridgeService } from './services/discord-bridge/discord-bridge.service';

describe('AppComponent', () => {
  let discordBridgeService: {
    indicator$: ReturnType<typeof of>;
    startStatusPolling: jest.Mock;
  };

  beforeEach(async () => {
    discordBridgeService = {
      indicator$: of({
        label: 'Connected',
        tone: 'ready',
        detail: '1 mapped channel ready',
        tooltip: 'Discord Gateway connected.',
        status: null,
      }),
      startStatusPolling: jest.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [AppComponent, NoopAnimationsModule],
      providers: [
        provideRouter([]),
        { provide: DiscordBridgeService, useValue: discordBridgeService },
      ]
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it(`should have the 'CORE UI' title`, () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app.title).toEqual('CORE UI');
  });

  it('should render the primary shell', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-side-navigation')).not.toBeNull();
    expect(compiled.querySelector('app-top-navigation')).not.toBeNull();
    expect(compiled.querySelector('router-outlet')).not.toBeNull();
  });

  it('should start Discord gateway status polling on init', () => {
    // Arrange
    const fixture = TestBed.createComponent(AppComponent);

    // Act
    fixture.detectChanges();

    // Assert
    expect(discordBridgeService.startStatusPolling).toHaveBeenCalled();
  });
});
