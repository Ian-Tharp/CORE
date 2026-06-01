import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { ConversationsPageComponent } from './conversations-page.component';

describe('ConversationsPageComponent', () => {
  let component: ConversationsPageComponent;
  let fixture: ComponentFixture<ConversationsPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConversationsPageComponent],
      // ConversationsPageComponent injects ConversationsService (HttpClient-based)
      // and renders child standalone components (ChatWindowComponent,
      // EnginePlaygroundComponent) whose root-provided services also depend on
      // HttpClient. The animations on the child chat window require an animations
      // provider. All of these are satisfied with the standard standalone testing
      // providers below.
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations()
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ConversationsPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
