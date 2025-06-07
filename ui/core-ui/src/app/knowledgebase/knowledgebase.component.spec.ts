import { ComponentFixture, TestBed } from '@angular/core/testing';

import { KnowledgebaseComponent } from './knowledgebase.component';

describe('KnowledgebaseComponent', () => {
  let component: KnowledgebaseComponent;
  let fixture: ComponentFixture<KnowledgebaseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [KnowledgebaseComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(KnowledgebaseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
