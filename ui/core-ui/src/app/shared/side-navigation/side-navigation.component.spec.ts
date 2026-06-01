import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { By } from '@angular/platform-browser';
import { provideRouter, RouterLink } from '@angular/router';
import { of } from 'rxjs';

import { SideNavigationComponent } from './side-navigation.component';
import { DiscordBridgeService } from '../../services/discord-bridge/discord-bridge.service';

describe('SideNavigationComponent', () => {
  let component: SideNavigationComponent;
  let fixture: ComponentFixture<SideNavigationComponent>;
  let discordBridgeService: { indicator$: ReturnType<typeof of> };

  beforeEach(async () => {
    discordBridgeService = {
      indicator$: of({
        label: 'Connected',
        tone: 'ready',
        detail: '1 mapped channel ready',
        tooltip: 'Discord Gateway connected.',
        status: null
      })
    };

    await TestBed.configureTestingModule({
      imports: [SideNavigationComponent, NoopAnimationsModule],
      providers: [
        provideRouter([]),
        { provide: DiscordBridgeService, useValue: discordBridgeService }
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(SideNavigationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should expose the tools route in primary navigation', () => {
    // Arrange / Act
    const routerLinks = fixture.debugElement
      .queryAll(By.directive(RouterLink))
      .map((debugElement) => debugElement.injector.get(RouterLink));

    // Assert
    expect(routerLinks.some((link) => link.urlTree?.toString() === '/tools')).toBe(true);
  });

  it('should render a Discord gateway status dot on the tools entry', () => {
    // Arrange / Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.querySelector('.nav-status-badge .status-dot.ready')).not.toBeNull();
  });
});
