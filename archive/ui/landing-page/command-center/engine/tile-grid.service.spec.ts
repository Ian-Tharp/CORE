import * as THREE from 'three';

import { EngineService } from './engine.service';
import { TileGridService } from './tile-grid.service';

describe('TileGridService', () => {
  let service: TileGridService;
  let pointerDownHandler: ((intersections: THREE.Intersection[], button: number) => void) | undefined;

  const createEngineMock = (): EngineService => ({
    add: jest.fn(),
    remove: jest.fn(),
    onHover: jest.fn(),
    onPointerDown: jest.fn((callback) => {
      pointerDownHandler = callback;
    }),
    onPointerUp: jest.fn(),
    onBeforeRender: jest.fn().mockReturnValue(jest.fn()),
    onContextClick: jest.fn(),
    recenterTo: jest.fn(),
    fitToBounds: jest.fn(),
    focusOn: jest.fn(),
    returnToOverview: jest.fn(),
    worldToCanvas: jest.fn()
  } as unknown as EngineService);

  const getInstancedMesh = (): THREE.InstancedMesh => (
    service as unknown as { instancedMesh: THREE.InstancedMesh }
  ).instancedMesh;

  const createHit = (mesh: THREE.InstancedMesh): THREE.Intersection => ({
    object: mesh,
    instanceId: 0,
    point: new THREE.Vector3()
  } as unknown as THREE.Intersection);

  beforeEach(() => {
    service = new TileGridService();
    pointerDownHandler = undefined;
    service.initialize(createEngineMock());
    service.createTileGrid({ cellRadius: 1, gridWidth: 1, gridHeight: 1, elevation: 0.1 });
  });

  it('should select without painting when view mode receives a tile click', () => {
    // Arrange
    let selectedIndex: number | null = null;
    service.setTerrainTool('water');
    service.setEditMode(false);
    service.onSelectedChanged().subscribe((selected) => {
      selectedIndex = selected?.index ?? null;
    });

    // Act
    pointerDownHandler?.([createHit(getInstancedMesh())], 0);
    const snapshot = service.snapshot('view-mode-world');

    // Assert
    expect(selectedIndex).toBe(0);
    expect(snapshot.layers.terrain).toEqual([]);
  });

  it('should paint without selecting when edit mode receives a tile click', () => {
    // Arrange
    let selectedIndex: number | null = null;
    service.setTerrainTool('water');
    service.setEditMode(true);
    service.onSelectedChanged().subscribe((selected) => {
      selectedIndex = selected?.index ?? null;
    });

    // Act
    pointerDownHandler?.([createHit(getInstancedMesh())], 0);
    const snapshot = service.snapshot('edit-mode-world');

    // Assert
    expect(selectedIndex).toBeNull();
    expect(snapshot.layers.terrain).toEqual([{ index: 0, state: 'water' }]);
  });
});
