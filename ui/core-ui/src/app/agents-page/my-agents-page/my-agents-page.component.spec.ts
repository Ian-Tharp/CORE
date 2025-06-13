import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MyAgentsPageComponent } from './my-agents-page.component';

describe('MyAgentsPageComponent', () => {
  let component: MyAgentsPageComponent;
  let fixture: ComponentFixture<MyAgentsPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MyAgentsPageComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MyAgentsPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
