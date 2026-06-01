import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { WorldsGridComponent } from './worlds-grid.component';

describe('WorldsGridComponent', () => {
  let component: WorldsGridComponent;
  let fixture: ComponentFixture<WorldsGridComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WorldsGridComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([])
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(WorldsGridComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
