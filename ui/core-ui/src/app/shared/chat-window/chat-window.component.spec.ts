import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { of } from 'rxjs';

import { ChatWindowComponent } from './chat-window.component';
import { ChatService } from '../../services/chat/chat-service';
import { KnowledgebaseService } from '../../services/knowledgebase/knowledgebase.service';

describe('ChatWindowComponent', () => {
  let component: ChatWindowComponent;
  let fixture: ComponentFixture<ChatWindowComponent>;

  const chatServiceMock = {
    sendMessage: jest.fn().mockReturnValue(of('')),
    clearMessages: jest.fn(),
    addMessage: jest.fn()
  };

  const kbServiceMock = {
    // Component subscribes to files$ in its constructor.
    files$: of([])
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      // ChatWindowComponent is standalone -> import it, do not declare it.
      imports: [ChatWindowComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideNoopAnimations(),
        { provide: ChatService, useValue: chatServiceMock },
        { provide: KnowledgebaseService, useValue: kbServiceMock }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(ChatWindowComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
