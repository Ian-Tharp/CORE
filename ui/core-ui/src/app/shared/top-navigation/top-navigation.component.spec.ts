import { ComponentFixture, TestBed } from '@angular/core/testing';
import { OverlayContainer } from '@angular/cdk/overlay';
import { provideRouter } from '@angular/router';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';

import { TopNavigationComponent } from './top-navigation.component';

describe('TopNavigationComponent', () => {
  let component: TopNavigationComponent;
  let fixture: ComponentFixture<TopNavigationComponent>;
  let overlayContainer: OverlayContainer;

  afterEach(() => {
    jest.restoreAllMocks();
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TopNavigationComponent, NoopAnimationsModule],
      providers: [provideRouter([])]
    })
      .compileComponents();

    fixture = TestBed.createComponent(TopNavigationComponent);
    component = fixture.componentInstance;
    overlayContainer = TestBed.inject(OverlayContainer);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should expose the tools hub in the system menu', async () => {
    // Arrange
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const navigateSpy = jest.spyOn(component, 'navigateTo');
    const systemButton = fixture.nativeElement.querySelector('button[aria-label="System"]');

    // Act
    systemButton.click();
    fixture.detectChanges();
    await fixture.whenStable();

    const overlayElement = overlayContainer.getContainerElement();
    const toolsButton = Array.from(overlayElement.querySelectorAll('button')).find((button) =>
      button.textContent?.includes('Tools & Integrations')
    ) as HTMLButtonElement | undefined;

    toolsButton?.click();

    // Assert
    expect(overlayElement.textContent).toContain('Tools & Integrations');
    expect(navigateSpy).toHaveBeenCalledWith('/tools');
  });
});
