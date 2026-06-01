import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { of } from 'rxjs';

import { CommandCenterComponent } from './command-center.component';
import { EngineService } from './engine/engine.service';
import { TileGridService } from './engine/tile-grid.service';
import { TileMetadataService } from './engine/tile-metadata.service';

// The component's ngAfterViewInit drives a real THREE.js/WebGL engine, which is
// unavailable in jsdom. Stub the engine-facing services so the component can be
// constructed and rendered without a GPU context, while leaving every other
// dependency (WorldsService -> HttpClient, ProjectService, UiNotifyService, etc.)
// resolved through real providers.
describe('CommandCenterComponent', () => {
  let component: CommandCenterComponent;
  let fixture: ComponentFixture<CommandCenterComponent>;

  const engineMock = {
    initialize: jest.fn(),
    start: jest.fn(),
    dispose: jest.fn()
  };

  const tileGridMock = {
    initialize: jest.fn(),
    onHoverChanged: jest.fn().mockReturnValue(of(null)),
    onSelectedChanged: jest.fn().mockReturnValue(of(null)),
    onTileContext: jest.fn(),
    createTileGrid: jest.fn(),
    updateConnections: jest.fn(),
    restore: jest.fn(),
    snapshot: jest.fn().mockReturnValue({ name: 'World', config: {}, layers: {} }),
    setActiveLayer: jest.fn(),
    setTerrainTool: jest.fn(),
    setBiomeTool: jest.fn(),
    setResourceTool: jest.fn(),
    setEditMode: jest.fn(),
    setLayerVisibility: jest.fn(),
    setOutlinesVisible: jest.fn(),
    setConnectionsVisible: jest.fn(),
    setBrushRadius: jest.fn(),
    setRandomSeed: jest.fn(),
    randomize: jest.fn(),
    clear: jest.fn(),
    getTileWorldPosition: jest.fn()
  };

  const tileMetadataMock = {
    onConnectionsChanged: jest.fn().mockReturnValue(of([])),
    getConnections: jest.fn().mockReturnValue([]),
    onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
    addConnection: jest.fn(),
    addAIObservation: jest.fn(),
    setSelectedTile: jest.fn(),
    filterByTag: jest.fn().mockReturnValue([]),
    getAllTags: jest.fn().mockReturnValue([]),
    getTilesWithContent: jest.fn().mockReturnValue([])
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CommandCenterComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        provideNoopAnimations(),
        { provide: EngineService, useValue: engineMock },
        { provide: TileGridService, useValue: tileGridMock },
        { provide: TileMetadataService, useValue: tileMetadataMock }
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(CommandCenterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
