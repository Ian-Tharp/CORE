import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { of, throwError } from 'rxjs';

import { WikiComponent } from './wiki.component';
import { CreativeDataService, WikiPage, WikiPageType } from '../services/creative-data.service';
import { CreativeService } from '../../services/creative/creative.service';

describe('WikiComponent', () => {
  let component: WikiComponent;
  let fixture: ComponentFixture<WikiComponent>;
  let mockCreativeDataService: jest.Mocked<CreativeDataService>;
  let mockCreativeService: jest.Mocked<CreativeService>;
  let mockActivatedRoute: jest.Mocked<ActivatedRoute>;

  const mockWikiPage: WikiPage = {
    id: 'test-id',
    worldId: 'world-1',
    title: 'Test Page',
    content: 'Test content',
    metadata: {
      type: 'Lore',
      tags: ['test', 'sample'],
      connections: []
    },
    media: [],
    createdAt: '2023-01-01T00:00:00.000Z',
    updatedAt: '2023-01-01T00:00:00.000Z'
  };

  beforeEach(async () => {
    const creativeDataServiceMock = {
      listWiki: jest.fn(),
      createWiki: jest.fn(),
      upsertWiki: jest.fn(),
      write: jest.fn()
    };
    const creativeServiceMock = {
      listWiki: jest.fn(),
      createWiki: jest.fn(),
      updateWiki: jest.fn(),
      createCharacter: jest.fn(),
      generateCharacterImage: jest.fn()
    };
    const activatedRouteMock = {
      snapshot: { queryParamMap: { get: jest.fn().mockReturnValue('test-world-id') } }
    };

    await TestBed.configureTestingModule({
      imports: [WikiComponent, FormsModule],
      providers: [
        { provide: CreativeDataService, useValue: creativeDataServiceMock },
        { provide: CreativeService, useValue: creativeServiceMock },
        { provide: ActivatedRoute, useValue: activatedRouteMock }
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WikiComponent);
    component = fixture.componentInstance;
    mockCreativeDataService = TestBed.inject(CreativeDataService) as jest.Mocked<CreativeDataService>;
    mockCreativeService = TestBed.inject(CreativeService) as jest.Mocked<CreativeService>;
    mockActivatedRoute = TestBed.inject(ActivatedRoute) as jest.Mocked<ActivatedRoute>;

    // Setup default return values
    mockCreativeService.listWiki.mockReturnValue(of([]));
    mockCreativeService.createWiki.mockReturnValue(of({ id: 'new-id' }));
    mockCreativeService.updateWiki.mockReturnValue(of({}));
    mockCreativeDataService.listWiki.mockReturnValue([]);
    mockCreativeDataService.createWiki.mockReturnValue(mockWikiPage);
  });

  describe('Component Initialization', () => {
    it('should create', () => {
      expect(component).toBeTruthy();
    });

    it('should initialize with default values', () => {
      expect(component.section).toBe('All');
      expect(component.viewMode).toBe('list');
      expect(component.searchQuery).toBe('');
      expect(component.selectedTags).toEqual([]);
      expect(component.showTemplateSelector).toBe(false);
      expect(component.contextMenuVisible).toBe(false);
    });

    it('should set worldId from route on init', () => {
      component.ngOnInit();
      expect(component.worldId).toBe('test-world-id');
    });
  });

  describe('Page Management', () => {
    beforeEach(() => {
      fixture.detectChanges();
    });

    it('should refresh pages from API successfully', () => {
      const mockPages = [mockWikiPage];
      mockCreativeService.listWiki.mockReturnValue(of(mockPages.map(p => ({
        ...p,
        world_id: p.worldId,
        created_at: p.createdAt,
        updated_at: p.updatedAt
      }))));

      component.refresh();

      expect(mockCreativeService.listWiki).toHaveBeenCalledWith('test-world-id');
      expect(component.pages.length).toBe(1);
      expect(component.pages[0].title).toBe('Test Page');
    });

    it('should fallback to local storage when API fails', () => {
      mockCreativeService.listWiki.mockReturnValue(throwError(() => 'API Error'));
      mockCreativeDataService.listWiki.mockReturnValue([mockWikiPage]);

      component.refresh();

      expect(mockCreativeDataService.listWiki).toHaveBeenCalledWith('test-world-id');
      expect(component.pages.length).toBe(1);
    });

    it('should create new page successfully', () => {
      component.newTitle = 'New Page';
      component.section = 'Lore';
      mockCreativeService.createWiki.mockReturnValue(of({ id: 'new-id' }));

      component.create();

      expect(mockCreativeDataService.createWiki).toHaveBeenCalledWith('test-world-id', 'New Page');
      expect(mockCreativeService.createWiki).toHaveBeenCalled();
      expect(component.editing).toBeTruthy();
      expect(component.newTitle).toBe('');
    });

    it('should not create page with empty title', () => {
      component.newTitle = '';
      component.create();

      expect(mockCreativeDataService.createWiki).not.toHaveBeenCalled();
    });

    it('should save page changes', () => {
      component.editing = { ...mockWikiPage };
      component.editing.title = 'Updated Title';

      component.save();

      expect(mockCreativeDataService.upsertWiki).toHaveBeenCalledWith(component.editing);
      expect(mockCreativeService.updateWiki).toHaveBeenCalled();
      expect(component.editing.updatedAt).toBeDefined();
    });
  });

  describe('Template System', () => {
    beforeEach(() => {
      fixture.detectChanges();
    });

    it('should create page from template', () => {
      component.createFromTemplate('Character');

      expect(mockCreativeDataService.createWiki).toHaveBeenCalled();
      expect(component.editing?.content).toContain('# Character Name');
      expect(component.editing?.metadata?.type).toBe('Lore');
      expect(component.editing?.metadata?.tags).toContain('character');
      expect(component.showTemplateSelector).toBe(false);
    });

    it('should generate content outline for page type', () => {
      component.editing = { ...mockWikiPage, metadata: { type: 'Factions' } };

      component.generateContentOutline();

      expect(component.editing.content).toContain('## Leadership');
      expect(component.editing.content).toContain('## Goals and Motivations');
    });

    it('should not generate outline without page type', () => {
      component.editing = { ...mockWikiPage, metadata: {} };
      const originalContent = component.editing.content;

      component.generateContentOutline();

      expect(component.editing.content).toBe(originalContent);
    });
  });

  describe('Filtering and Search', () => {
    beforeEach(() => {
      component.pages = [
        { ...mockWikiPage, title: 'Lore Page', metadata: { type: 'Lore', tags: ['history'] } },
        { ...mockWikiPage, id: '2', title: 'Faction Page', metadata: { type: 'Factions', tags: ['military'] } },
        { ...mockWikiPage, id: '3', title: 'Search Test', content: 'Contains searchable text' }
      ];
      fixture.detectChanges();
    });

    it('should filter pages by section', () => {
      component.section = 'Lore';
      const filtered = component.filteredPages();
      expect(filtered.length).toBe(1);
      expect(filtered[0].title).toBe('Lore Page');
    });

    it('should return all pages when section is All', () => {
      component.section = 'All';
      const filtered = component.filteredPages();
      expect(filtered.length).toBe(3);
    });

    it('should filter pages by search query in title', () => {
      component.searchQuery = 'Search';
      const filtered = component.filteredPages();
      expect(filtered.length).toBe(1);
      expect(filtered[0].title).toBe('Search Test');
    });

    it('should filter pages by search query in content', () => {
      component.searchQuery = 'searchable';
      const filtered = component.filteredPages();
      expect(filtered.length).toBe(1);
      expect(filtered[0].title).toBe('Search Test');
    });

    it('should filter pages by selected tags', () => {
      component.selectedTags = ['military'];
      const filtered = component.filteredPages();
      expect(filtered.length).toBe(1);
      expect(filtered[0].title).toBe('Faction Page');
    });
  });

  describe('Tag Management', () => {
    beforeEach(() => {
      component.editing = { ...mockWikiPage };
      fixture.detectChanges();
    });

    it('should add new tag', () => {
      component.addTag('newtag');
      expect(component.editing?.metadata?.tags).toContain('newtag');
    });

    it('should not add duplicate tag', () => {
      component.addTag('test'); // 'test' already exists in mockWikiPage
      expect(component.editing?.metadata?.tags?.filter(tag => tag === 'test').length).toBe(1);
    });

    it('should not add empty tag', () => {
      const initialTagCount = component.editing?.metadata?.tags?.length || 0;
      component.addTag('');
      expect(component.editing?.metadata?.tags?.length).toBe(initialTagCount);
    });

    it('should remove tag', () => {
      component.removeTag('test');
      expect(component.editing?.metadata?.tags).not.toContain('test');
    });

    it('should get all unique tags from pages', () => {
      component.pages = [
        { ...mockWikiPage, metadata: { tags: ['tag1', 'tag2'] } },
        { ...mockWikiPage, id: '2', metadata: { tags: ['tag2', 'tag3'] } }
      ];
      const allTags = component.getAllTags();
      expect(allTags).toEqual(['tag1', 'tag2', 'tag3']);
    });

    it('should toggle tag selection', () => {
      component.toggleTag('newtag');
      expect(component.selectedTags).toContain('newtag');

      component.toggleTag('newtag');
      expect(component.selectedTags).not.toContain('newtag');
    });
  });

  describe('Context Menu', () => {
    beforeEach(() => {
      fixture.detectChanges();
    });

    it('should show context menu', () => {
      const mockEvent = new MouseEvent('contextmenu', { clientX: 100, clientY: 200 });
      const preventDefaultSpy = jest.spyOn(mockEvent, 'preventDefault');
      const stopPropagationSpy = jest.spyOn(mockEvent, 'stopPropagation');

      component.showContextMenu(mockEvent, mockWikiPage);

      expect(preventDefaultSpy).toHaveBeenCalled();
      expect(stopPropagationSpy).toHaveBeenCalled();
      expect(component.contextMenuVisible).toBe(true);
      expect(component.contextMenuX).toBe(100);
      expect(component.contextMenuY).toBe(200);
      expect(component.contextPage).toBe(mockWikiPage);
    });

    it('should hide context menu', () => {
      component.contextMenuVisible = true;
      component.contextPage = mockWikiPage;

      component.hideContextMenu();

      expect(component.contextMenuVisible).toBe(false);
      expect(component.contextPage).toBeUndefined();
    });

    it('should duplicate page', () => {
      component.contextPage = mockWikiPage;

      component.duplicatePage();

      expect(mockCreativeDataService.createWiki).toHaveBeenCalledWith('world-1', 'Test Page (Copy)');
      expect(mockCreativeService.createWiki).toHaveBeenCalled();
    });

    it('should start editing page title', () => {
      component.contextPage = mockWikiPage;

      component.editPageTitle();

      expect(component.editingTitle).toBe(true);
      expect(component.tempTitle).toBe('Test Page');
      expect(component.contextMenuVisible).toBe(false);
    });

    it('should confirm title edit and update page', () => {
      component.contextPage = mockWikiPage;
      component.pages = [mockWikiPage];
      component.tempTitle = 'New Title';
      component.editingTitle = true;

      component.confirmTitleEdit();

      expect(component.editingTitle).toBe(false);
      expect(component.tempTitle).toBe('');
      expect(component.pages[0].title).toBe('New Title');
      expect(mockCreativeDataService.upsertWiki).toHaveBeenCalled();
      expect(mockCreativeService.updateWiki).toHaveBeenCalled();
    });

    it('should cancel title edit without changes', () => {
      component.tempTitle = 'Some Title';
      component.editingTitle = true;

      component.cancelTitleEdit();

      expect(component.editingTitle).toBe(false);
      expect(component.tempTitle).toBe('');
    });

    it('should cancel title edit when empty title provided', () => {
      component.contextPage = mockWikiPage;
      component.tempTitle = '';
      component.editingTitle = true;

      const cancelSpy = jest.spyOn(component, 'cancelTitleEdit');
      component.confirmTitleEdit();

      expect(cancelSpy).toHaveBeenCalled();
    });
  });

  describe('Media Management', () => {
    beforeEach(() => {
      component.editing = { ...mockWikiPage };
      fixture.detectChanges();
    });

    it('should handle file upload', () => {
      const mockFile = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
      const mockEvent = { target: { files: [mockFile] } } as any;

      const uploadMediaSpy = jest.spyOn(component, 'uploadMedia');
      component.onFileSelected(mockEvent);

      expect(uploadMediaSpy).toHaveBeenCalledWith(mockFile);
    });

    it('should handle drag over', () => {
      const mockEvent = new DragEvent('dragover');
      const preventDefaultSpy = jest.spyOn(mockEvent, 'preventDefault');

      component.onDragOver(mockEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
      expect(component.dragOverEditor).toBe(true);
    });

    it('should handle drag leave', () => {
      const mockEvent = new DragEvent('dragleave');
      const preventDefaultSpy = jest.spyOn(mockEvent, 'preventDefault');
      component.dragOverEditor = true;

      component.onDragLeave(mockEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
      expect(component.dragOverEditor).toBe(false);
    });

    it('should remove media item', () => {
      component.editing!.media = [
        { id: 'media-1', url: 'test1.jpg', type: 'image' },
        { id: 'media-2', url: 'test2.jpg', type: 'image' }
      ];

      component.removeMedia('media-1');

      expect(component.editing!.media?.length).toBe(1);
      expect(component.editing!.media?.[0].id).toBe('media-2');
    });
  });

  describe('AI Features', () => {
    beforeEach(() => {
      component.editing = { ...mockWikiPage };
      fixture.detectChanges();
    });

    it('should get content suggestions for page type', () => {
      component.editing!.metadata!.type = 'Lore';
      const suggestions = component.getContentSuggestions();

      expect(suggestions.length).toBeGreaterThan(0);
      expect(suggestions[0]).toContain('historical timeline');
    });

    it('should return empty suggestions without page type', () => {
      component.editing!.metadata = {};
      const suggestions = component.getContentSuggestions();

      expect(suggestions.length).toBe(0);
    });

    it('should get recommended tags for page type', () => {
      component.editing!.metadata = { type: 'Biomes', tags: [] };
      const recommendedTags = component.getRecommendedTags();

      expect(recommendedTags).toContain('environment');
      expect(recommendedTags).toContain('climate');
    });

    it('should filter out existing tags from recommendations', () => {
      component.editing!.metadata = { type: 'Biomes', tags: ['environment'] };
      const recommendedTags = component.getRecommendedTags();

      expect(recommendedTags).not.toContain('environment');
    });

    it('should apply content suggestion', () => {
      const originalContent = component.editing!.content;
      component.applySuggestion('Add historical timeline with key events');

      expect(component.editing!.content).not.toBe(originalContent);
      expect(component.editing!.content).toContain('historical timeline');
    });

    it('should get connection suggestions', () => {
      component.pages = [
        { ...mockWikiPage, title: 'Related Page', metadata: { tags: ['test'] } },
        { ...mockWikiPage, id: '2', title: 'Unrelated Page', metadata: { tags: ['other'] } }
      ];

      const connections = component.getConnectionSuggestions();
      expect(connections.length).toBe(1);
      expect(connections[0].title).toBe('Related Page');
    });

    it('should connect pages bidirectionally', () => {
      const targetPage = { ...mockWikiPage, id: 'target-id', metadata: { connections: [] } };
      component.connectToPage(targetPage);

      expect(component.editing!.metadata!.connections).toContain('target-id');
      expect(mockCreativeDataService.upsertWiki).toHaveBeenCalledWith(targetPage);
    });
  });

  describe('Character Builder', () => {
    beforeEach(() => {
      fixture.detectChanges();
    });

    it('should create character with valid input', () => {
      component.characterName = 'Test Character';
      component.characterTraits = '{"role":"Warrior","mood":"Brave"}';
      mockCreativeService.createCharacter.mockReturnValue(of({ id: 'char-1' }));
      mockCreativeService.generateCharacterImage.mockReturnValue(of({}));

      component.createCharacter();

      expect(mockCreativeService.createCharacter).toHaveBeenCalledWith({
        world_id: 'test-world-id',
        name: 'Test Character',
        traits: { role: 'Warrior', mood: 'Brave' }
      });
      expect(component.characterName).toBe('');
      expect(component.characterTraits).toBe('');
    });

    it('should handle invalid JSON in character traits', () => {
      component.characterName = 'Test Character';
      component.characterTraits = 'invalid json';
      mockCreativeService.createCharacter.mockReturnValue(of({ id: 'char-1' }));
      mockCreativeService.generateCharacterImage.mockReturnValue(of({}));

      component.createCharacter();

      expect(mockCreativeService.createCharacter).toHaveBeenCalledWith({
        world_id: 'test-world-id',
        name: 'Test Character',
        traits: { description: 'invalid json' }
      });
    });

    it('should not create character without name', () => {
      component.characterName = '';
      component.createCharacter();

      expect(mockCreativeService.createCharacter).not.toHaveBeenCalled();
    });
  });

  describe('Utility Methods', () => {
    it('should safely get tags from page', () => {
      const pageWithTags = { ...mockWikiPage, metadata: { tags: ['tag1', 'tag2'] } };
      const pageWithoutTags = { ...mockWikiPage, metadata: {} };

      expect(component.safeGetTags(pageWithTags)).toEqual(['tag1', 'tag2']);
      expect(component.safeGetTags(pageWithoutTags)).toEqual([]);
    });

    it('should safely get media URL', () => {
      const pageWithMedia = {
        ...mockWikiPage,
        media: [{ id: '1', url: 'test.jpg', type: 'image' as const }]
      };
      const pageWithoutMedia = { ...mockWikiPage, media: [] };

      expect(component.safeGetMediaUrl(pageWithMedia, 0)).toBe('test.jpg');
      expect(component.safeGetMediaUrl(pageWithoutMedia, 0)).toBeUndefined();
    });

    it('should safely check if page has media', () => {
      const pageWithMedia = {
        ...mockWikiPage,
        media: [{ id: '1', url: 'test.jpg', type: 'image' as const }]
      };
      const pageWithoutMedia = { ...mockWikiPage, media: [] };

      expect(component.safeHasMedia(pageWithMedia)).toBe(true);
      expect(component.safeHasMedia(pageWithoutMedia)).toBe(false);
    });
  });
});
