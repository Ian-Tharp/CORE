import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { WorldDetailComponent } from './world-detail.component';
import { ProjectService } from '../../landing-page/command-center/engine/project.service';

describe('WorldDetailComponent', () => {
  let component: WorldDetailComponent;
  let fixture: ComponentFixture<WorldDetailComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WorldDetailComponent],
      providers: [
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: {
            snapshot: { paramMap: convertToParamMap({ id: 'world-1' }) },
            params: of({ id: 'world-1' }),
            paramMap: of(convertToParamMap({ id: 'world-1' }))
          }
        },
        {
          provide: ProjectService,
          useValue: {
            load: jest.fn().mockReturnValue(undefined)
          }
        }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(WorldDetailComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
