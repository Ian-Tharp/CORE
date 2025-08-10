import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreativeBoardsComponent } from './creative-boards.component';

describe('CreativeBoardsComponent', () => {
  let component: CreativeBoardsComponent;
  let fixture: ComponentFixture<CreativeBoardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreativeBoardsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreativeBoardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
