import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  provideHttpClientTesting,
  HttpTestingController
} from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { MarketplaceComponent } from './marketplace.component';

describe('MarketplaceComponent', () => {
  let component: MarketplaceComponent;
  let fixture: ComponentFixture<MarketplaceComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MarketplaceComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        provideNoopAnimations()
      ]
    }).compileComponents();

    httpMock = TestBed.inject(HttpTestingController);

    fixture = TestBed.createComponent(MarketplaceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();

    // ngOnInit -> refresh() -> WorldsService.listWorlds() issues a GET.
    // Flush it with an empty list so no real backend is contacted.
    const req = httpMock.expectOne((r) => r.url === 'http://localhost:8001/worlds');
    req.flush([]);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
