import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { of } from 'rxjs';

import { BoardsComponent } from './boards.component';
import { ChatService } from '../../services/chat/chat-service';
import { KnowledgebaseService } from '../../services/knowledgebase/knowledgebase.service';

describe('BoardsComponent', () => {
  let component: BoardsComponent;
  let fixture: ComponentFixture<BoardsComponent>;

  // Mocks for the services the embedded ChatWindowComponent injects.
  const chatServiceMock = {
    messages$: of([]),
    addMessage: jest.fn(),
    clearMessages: jest.fn(),
    sendMessage: jest.fn().mockReturnValue(of(''))
  };

  const knowledgebaseServiceMock = {
    files$: of([])
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BoardsComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations(),
        { provide: ChatService, useValue: chatServiceMock },
        { provide: KnowledgebaseService, useValue: knowledgebaseServiceMock }
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(BoardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    // Arrange
    // Act
    fixture.detectChanges();

    // Assert
    expect(component).toBeTruthy();
  });

  it('should render the calendar, task, event, and stats sections', () => {
    // Arrange
    const compiled = fixture.nativeElement as HTMLElement;

    // Act
    fixture.detectChanges();

    // Assert
    expect(compiled.querySelector('.calendar-card')?.textContent).toContain('Task Calendar');
    expect(compiled.querySelector('.tasks-card')?.textContent).toContain('Tasks for');
    expect(compiled.querySelector('.events-card')?.textContent).toContain('Upcoming Events');
    expect(compiled.querySelector('.stats-card')?.textContent).toContain('Quick Stats');
  });

  it('should collapse the chat panel from the toggle button', () => {
    // Arrange
    const compiled = fixture.nativeElement as HTMLElement;
    const toggleButton = compiled.querySelector<HTMLButtonElement>('.chat-section__toggle');

    // Act
    toggleButton?.click();
    fixture.detectChanges();

    // Assert
    expect(compiled.querySelector('.boards-layout')?.classList).toContain('boards-layout--chat-collapsed');
    expect(compiled.querySelector('.chat-section')?.textContent).toContain('Chat');
  });

  it('should keep the existing selected date when the calendar emits null', () => {
    // Arrange
    const selectedDate = component.selectedDate;

    // Act
    component.onDateSelected(null);

    // Assert
    expect(component.selectedDate).toBe(selectedDate);
  });

  it('should show the maintenance icon for maintenance events', () => {
    // Arrange
    const eventType = 'maintenance';

    // Act
    const icon = component.getEventTypeIcon(eventType);

    // Assert
    expect(icon).toBe('build');
  });
});
