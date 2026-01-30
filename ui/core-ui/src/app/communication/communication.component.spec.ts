import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { MessageService } from './services/message.service';

import { CommunicationComponent } from './communication.component';

describe('CommunicationComponent', () => {
  let component: CommunicationComponent;
  let fixture: ComponentFixture<CommunicationComponent>;

  beforeEach(async () => {
    const messageSvcMock = {
      sendMessage: jest.fn().mockReturnValue(of({}))
    } as any as MessageService;

    await TestBed.configureTestingModule({
      imports: [CommunicationComponent, HttpClientTestingModule],
      providers: [
        { provide: MessageService, useValue: messageSvcMock }
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CommunicationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('prefills @mention on reply and sends parent/thread', () => {
    // Arrange
    component.selectedChannel = {
      channel_id: 'c1', channel_type: 'team', name: 'Team 1', is_persistent: true, is_public: true, created_at: new Date().toISOString()
    } as any;
    const parent: any = {
      message_id: 'm1', sender_name: 'Threshold', sender_id: 'instance_011_threshold', sender_type: 'agent', channel_id: 'c1'
    };

    // Act
    component.replyToMessage(parent);

    // Assert prefill
    expect(component.messageText.startsWith('@Threshold ')).toBeTruthy();

    // Act send
    component.messageText += 'Hello there';
    (TestBed.inject(MessageService) as any).sendMessage.mockClear();
    component.sendMessage();

    // Assert parent/thread passed
    const args = (TestBed.inject(MessageService) as any).sendMessage.mock.calls[0];
    expect(args[0]).toBe('c1');
    expect(args[1]).toContain('Hello');
    // parent_message_id
    expect(args[4]).toBe('m1');
    // thread_id (root = parent when absent)
    expect(args[5]).toBe('m1');
  });
});
