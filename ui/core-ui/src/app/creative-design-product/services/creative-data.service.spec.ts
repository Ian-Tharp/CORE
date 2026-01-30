import { TestBed } from '@angular/core/testing';
import { CreativeDataService, WikiPage, Board, WikiPageType } from './creative-data.service';

// Jest setup for localStorage mock
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: jest.fn((key: string) => { delete store[key]; }),
    clear: jest.fn(() => { store = {}; })
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

describe('CreativeDataService', () => {
  let service: CreativeDataService;

  const mockWikiPage: WikiPage = {
    id: 'wiki-1',
    worldId: 'world-1',
    title: 'Test Wiki Page',
    content: 'Test content',
    richContent: null,
    media: [],
    metadata: {
      type: 'Lore',
      tags: ['test', 'sample'],
      connections: [],
      icon: '📄'
    },
    createdAt: '2023-01-01T00:00:00.000Z',
    updatedAt: '2023-01-01T00:00:00.000Z'
  };

  const mockBoard: Board = {
    id: 'board-1',
    worldId: 'world-1',
    title: 'Test Board',
    cards: [],
    createdAt: '2023-01-01T00:00:00.000Z'
  };

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CreativeDataService);

    // Clear localStorage mock before each test
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  afterEach(() => {
    // Clean up localStorage mock after each test
    localStorageMock.clear();
  });

  describe('Service Initialization', () => {
    it('should be created', () => {
      expect(service).toBeTruthy();
    });
  });

  describe('Board Management', () => {
    describe('listBoards', () => {
      it('should return empty array when no boards exist', () => {
        const boards = service.listBoards();
        expect(boards).toEqual([]);
      });

      it('should return all boards when no worldId specified', () => {
        // Setup test data
        const testBoards = [mockBoard, { ...mockBoard, id: 'board-2', worldId: 'world-2' }];
        localStorage.setItem('creative.boards.v1', JSON.stringify(testBoards));

        const boards = service.listBoards();
        expect(boards).toEqual(testBoards);
        expect(boards.length).toBe(2);
      });

      it('should filter boards by worldId', () => {
        // Setup test data
        const testBoards = [
          mockBoard,
          { ...mockBoard, id: 'board-2', worldId: 'world-2' },
          { ...mockBoard, id: 'board-3', worldId: 'world-1' }
        ];
        localStorage.setItem('creative.boards.v1', JSON.stringify(testBoards));

        const boards = service.listBoards('world-1');
        expect(boards.length).toBe(2);
        expect(boards.every(b => b.worldId === 'world-1')).toBe(true);
      });

      it('should handle malformed JSON in localStorage', () => {
        localStorage.setItem('creative.boards.v1', 'invalid json');

        const boards = service.listBoards();
        expect(boards).toEqual([]);
      });
    });

    describe('createBoard', () => {
      it('should create new board with required fields', () => {
        const boardData = { title: 'New Board', worldId: 'world-1' };
        const createdBoard = service.createBoard(boardData);

        expect(createdBoard.title).toBe('New Board');
        expect(createdBoard.worldId).toBe('world-1');
        expect(createdBoard.id).toBeDefined();
        expect(createdBoard.cards).toEqual([]);
        expect(createdBoard.createdAt).toBeDefined();
      });

      it('should persist board to localStorage', () => {
        const boardData = { title: 'New Board', worldId: 'world-1' };
        const createdBoard = service.createBoard(boardData);

        const storedBoards = service.listBoards();
        expect(storedBoards).toContain(createdBoard);
        expect(storedBoards.length).toBe(1);
      });

      it('should create board without worldId', () => {
        const boardData = { title: 'Global Board' };
        const createdBoard = service.createBoard(boardData);

        expect(createdBoard.title).toBe('Global Board');
        expect(createdBoard.worldId).toBeUndefined();
      });
    });
  });

  describe('Wiki Management', () => {
    describe('listWiki', () => {
      it('should return empty array when no pages exist', () => {
        const pages = service.listWiki();
        expect(pages).toEqual([]);
      });

      it('should return all pages when no worldId specified', () => {
        // Setup test data
        const testPages = [mockWikiPage, { ...mockWikiPage, id: 'wiki-2', worldId: 'world-2' }];
        localStorage.setItem('creative.wiki.v1', JSON.stringify(testPages));

        const pages = service.listWiki();
        expect(pages).toEqual(testPages);
        expect(pages.length).toBe(2);
      });

      it('should filter pages by worldId', () => {
        // Setup test data
        const testPages = [
          mockWikiPage,
          { ...mockWikiPage, id: 'wiki-2', worldId: 'world-2' },
          { ...mockWikiPage, id: 'wiki-3', worldId: 'world-1' }
        ];
        localStorage.setItem('creative.wiki.v1', JSON.stringify(testPages));

        const pages = service.listWiki('world-1');
        expect(pages.length).toBe(2);
        expect(pages.every(p => p.worldId === 'world-1')).toBe(true);
      });

      it('should handle malformed JSON in localStorage', () => {
        localStorage.setItem('creative.wiki.v1', 'invalid json');

        const pages = service.listWiki();
        expect(pages).toEqual([]);
      });
    });

    describe('createWiki', () => {
      it('should create new wiki page with required fields', () => {
        const createdPage = service.createWiki('world-1', 'Test Page');

        expect(createdPage.title).toBe('Test Page');
        expect(createdPage.worldId).toBe('world-1');
        expect(createdPage.id).toBeDefined();
        expect(createdPage.content).toBe('');
        expect(createdPage.richContent).toBeNull();
        expect(createdPage.media).toEqual([]);
        expect(createdPage.metadata).toEqual({ tags: [], connections: [] });
        expect(createdPage.createdAt).toBeDefined();
        expect(createdPage.updatedAt).toBeDefined();
      });

      it('should create page without worldId', () => {
        const createdPage = service.createWiki(undefined, 'Global Page');

        expect(createdPage.title).toBe('Global Page');
        expect(createdPage.worldId).toBeUndefined();
      });

      it('should automatically save created page to localStorage', () => {
        const createdPage = service.createWiki('world-1', 'Test Page');
        const storedPages = service.listWiki();

        expect(storedPages).toContain(createdPage);
        expect(storedPages.length).toBe(1);
      });

      it('should generate unique IDs for each page', () => {
        const page1 = service.createWiki('world-1', 'Page 1');
        const page2 = service.createWiki('world-1', 'Page 2');

        expect(page1.id).not.toBe(page2.id);
        expect(page1.id).toBeDefined();
        expect(page2.id).toBeDefined();
      });
    });

    describe('upsertWiki', () => {
      it('should add new page to storage', () => {
        service.upsertWiki(mockWikiPage);

        const storedPages = service.listWiki();
        expect(storedPages).toContain(mockWikiPage);
        expect(storedPages.length).toBe(1);
      });

      it('should update existing page', () => {
        // First, add a page
        service.upsertWiki(mockWikiPage);

        // Then update it
        const updatedPage = { ...mockWikiPage, title: 'Updated Title', content: 'Updated content' };
        service.upsertWiki(updatedPage);

        const storedPages = service.listWiki();
        expect(storedPages.length).toBe(1);
        expect(storedPages[0].title).toBe('Updated Title');
        expect(storedPages[0].content).toBe('Updated content');
      });

      it('should preserve other pages when updating', () => {
        const page1 = { ...mockWikiPage, id: 'page-1' };
        const page2 = { ...mockWikiPage, id: 'page-2', title: 'Page 2' };

        service.upsertWiki(page1);
        service.upsertWiki(page2);

        // Update page1
        const updatedPage1 = { ...page1, title: 'Updated Page 1' };
        service.upsertWiki(updatedPage1);

        const storedPages = service.listWiki();
        expect(storedPages.length).toBe(2);
        expect(storedPages.find(p => p.id === 'page-1')?.title).toBe('Updated Page 1');
        expect(storedPages.find(p => p.id === 'page-2')?.title).toBe('Page 2');
      });
    });
  });

  describe('Generic Storage Methods', () => {
    describe('write', () => {
      it('should write data to localStorage', () => {
        const testData = [{ id: '1', name: 'test' }];
        service.write('test.key', testData);

        const storedData = JSON.parse(localStorage.getItem('test.key') || '[]');
        expect(storedData).toEqual(testData);
      });

      it('should overwrite existing data', () => {
        const initialData = [{ id: '1', name: 'initial' }];
        const newData = [{ id: '1', name: 'updated' }, { id: '2', name: 'new' }];

        service.write('test.key', initialData);
        service.write('test.key', newData);

        const storedData = JSON.parse(localStorage.getItem('test.key') || '[]');
        expect(storedData).toEqual(newData);
        expect(storedData.length).toBe(2);
      });

      it('should handle empty arrays', () => {
        service.write('test.key', []);

        const storedData = JSON.parse(localStorage.getItem('test.key') || 'null');
        expect(storedData).toEqual([]);
      });
    });
  });

  describe('Data Integrity', () => {
    it('should maintain data consistency across board operations', () => {
      // Create multiple boards
      const board1 = service.createBoard({ title: 'Board 1', worldId: 'world-1' });
      const board2 = service.createBoard({ title: 'Board 2', worldId: 'world-2' });

      // Verify they are stored correctly
      const allBoards = service.listBoards();
      expect(allBoards.length).toBe(2);

      // Verify filtering works
      const world1Boards = service.listBoards('world-1');
      expect(world1Boards.length).toBe(1);
      expect(world1Boards[0].id).toBe(board1.id);
    });

    it('should maintain data consistency across wiki operations', () => {
      // Create multiple pages
      const page1 = service.createWiki('world-1', 'Page 1');
      const page2 = service.createWiki('world-2', 'Page 2');

      // Update one page
      const updatedPage1 = { ...page1, content: 'Updated content' };
      service.upsertWiki(updatedPage1);

      // Verify consistency
      const allPages = service.listWiki();
      expect(allPages.length).toBe(2);

      const world1Pages = service.listWiki('world-1');
      expect(world1Pages.length).toBe(1);
      expect(world1Pages[0].content).toBe('Updated content');

      const world2Pages = service.listWiki('world-2');
      expect(world2Pages.length).toBe(1);
      expect(world2Pages[0].title).toBe('Page 2');
    });

    it('should handle concurrent operations correctly', () => {
      // Simulate concurrent page creation
      const pages = [];
      for (let i = 0; i < 5; i++) {
        pages.push(service.createWiki('world-1', `Page ${i}`));
      }

      // Verify all pages were created with unique IDs
      const storedPages = service.listWiki('world-1');
      expect(storedPages.length).toBe(5);

      const ids = storedPages.map(p => p.id);
      const uniqueIds = [...new Set(ids)];
      expect(uniqueIds.length).toBe(5); // All IDs should be unique
    });
  });

  describe('Edge Cases', () => {
    it('should handle null/undefined localStorage values', () => {
      localStorage.removeItem('creative.wiki.v1');

      const pages = service.listWiki();
      expect(pages).toEqual([]);
    });

    it('should handle corrupted localStorage data', () => {
      localStorage.setItem('creative.wiki.v1', '{"incomplete": json');

      const pages = service.listWiki();
      expect(pages).toEqual([]);
    });

    it('should handle non-array data in localStorage', () => {
      localStorage.setItem('creative.wiki.v1', '{"not": "an array"}');

      const pages = service.listWiki();
      expect(pages).toEqual([]);
    });

    it('should handle empty string titles gracefully', () => {
      const page = service.createWiki('world-1', '');

      expect(page.title).toBe('');
      expect(page.id).toBeDefined();
    });

    it('should handle special characters in titles', () => {
      const specialTitle = 'Page with 特殊字符 and émojis 🌟';
      const page = service.createWiki('world-1', specialTitle);

      expect(page.title).toBe(specialTitle);

      const storedPages = service.listWiki();
      expect(storedPages[0].title).toBe(specialTitle);
    });
  });

  describe('Performance Considerations', () => {
    it('should handle large datasets efficiently', () => {
      // Create a large number of pages
      const startTime = performance.now();

      for (let i = 0; i < 100; i++) {
        service.createWiki('world-1', `Page ${i}`);
      }

      const creationTime = performance.now() - startTime;
      expect(creationTime).toBeLessThan(1000); // Should complete in under 1 second

      // Test retrieval performance
      const retrievalStartTime = performance.now();
      const pages = service.listWiki('world-1');
      const retrievalTime = performance.now() - retrievalStartTime;

      expect(pages.length).toBe(100);
      expect(retrievalTime).toBeLessThan(100); // Should retrieve in under 100ms
    });

    it('should handle filtering efficiently with large datasets', () => {
      // Create pages in multiple worlds
      for (let i = 0; i < 50; i++) {
        service.createWiki('world-1', `Page ${i}`);
        service.createWiki('world-2', `Page ${i}`);
      }

      const startTime = performance.now();
      const world1Pages = service.listWiki('world-1');
      const filterTime = performance.now() - startTime;

      expect(world1Pages.length).toBe(50);
      expect(filterTime).toBeLessThan(50); // Filtering should be fast
      expect(world1Pages.every(p => p.worldId === 'world-1')).toBe(true);
    });
  });
});