import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConversationsPageComponent } from './conversations-page.component';

describe('ConversationsPageComponent', () => {
  let component: ConversationsPageComponent;
  let fixture: ComponentFixture<ConversationsPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ConversationsPageComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ConversationsPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
