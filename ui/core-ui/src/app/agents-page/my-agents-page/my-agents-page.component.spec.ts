import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { MyAgentsPageComponent } from './my-agents-page.component';

describe('MyAgentsPageComponent', () => {
  let component: MyAgentsPageComponent;
  let fixture: ComponentFixture<MyAgentsPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MyAgentsPageComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        provideNoopAnimations()
      ]
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
