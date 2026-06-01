import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap } from '@angular/router';
import { of } from 'rxjs';

import { CreativeBoardsComponent } from './creative-boards.component';
import { CreativeDataService } from '../services/creative-data.service';

describe('CreativeBoardsComponent', () => {
  let component: CreativeBoardsComponent;
  let fixture: ComponentFixture<CreativeBoardsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreativeBoardsComponent],
      providers: [
        {
          provide: CreativeDataService,
          useValue: {
            listBoards: jest.fn().mockReturnValue([]),
            createBoard: jest.fn().mockReturnValue({
              id: 'b1',
              title: 'Board',
              cards: [],
              createdAt: new Date().toISOString()
            })
          }
        },
        {
          provide: ActivatedRoute,
          useValue: {
            snapshot: { queryParamMap: convertToParamMap({}) },
            params: of({}),
            queryParamMap: of(convertToParamMap({}))
          }
        }
      ]
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
