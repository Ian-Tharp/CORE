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
    expect(component).toBeTruthy();
  });
});
