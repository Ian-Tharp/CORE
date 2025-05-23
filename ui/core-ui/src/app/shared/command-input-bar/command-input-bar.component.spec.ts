import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CommandInputBarComponent } from './command-input-bar.component';

describe('CommandInputBarComponent', () => {
  let component: CommandInputBarComponent;
  let fixture: ComponentFixture<CommandInputBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CommandInputBarComponent]
    })
      .compileComponents();

    fixture = TestBed.createComponent(CommandInputBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
