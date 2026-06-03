import { expect, test } from '@playwright/test';

test.describe('Command Center world agents', () => {
  let loreWorkflowRequested = false;

  test.beforeEach(async ({ page }) => {
    loreWorkflowRequested = false;
    await page.addInitScript(() => {
      sessionStorage.setItem('core.universePicker.seen', '1');
    });

    await page.route('**/worlds/world-1/snapshots/latest', async (route) => {
      await route.fulfill({
        json: {
          id: 'snap-1',
          created_at: new Date().toISOString(),
          config: { radius: 1.2, gridWidth: 3, gridHeight: 3, elevation: 0.1 },
          layers: {
            terrain: [{ index: 0, state: 'water' }],
            biome: [{ index: 0, state: 'forest' }],
            resources: []
          },
          preview: null
        }
      });
    });
    await page.route('**/worlds/world-1/metadata', async (route) => {
      await route.fulfill({ json: { metadata: {}, connections: [] } });
    });
    await page.route('**/worlds/world-1/assets**', async (route) => {
      await route.fulfill({
        json: [{
          id: 'asset-1',
          tile_index: 0,
          kind: 'art',
          title: 'World portrait',
          image_b64: 'iVBORw0KGgo=',
          created_at: new Date().toISOString()
        }]
      });
    });
    await page.route('**/worlds/world-1/knowledge', async (route) => {
      await route.fulfill({ json: [] });
    });
    await page.route('**/creative/characters**', async (route) => {
      await route.fulfill({
        json: [{
          id: 'char-1',
          world_id: 'world-1',
          name: 'Kael Dawnseer',
          traits: { origin: 'ai' },
          image_b64: 'iVBORw0KGgo=',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }]
      });
    });
    await page.route('**/worlds/**/agents/lore/save', async (route) => {
      await route.fulfill({ json: { id: 'wiki-1', title: 'Verdant Gate' } });
    });
    await page.route('**/worlds/**/agents/lore', async (route) => {
      loreWorkflowRequested = true;
      await route.fulfill({
        json: {
          id: 'wiki-1',
          title: 'Verdant Gate',
          content: '# Verdant Gate\n\nA forested water world with living causeways.',
          generated_by: 'world_lore_architect',
          audit: {
            approved: true,
            confidence: 0.92,
            contradictions: [],
            missing_details: [],
            suggestions: []
          }
        }
      });
    });
  });

  test('generates lore through the modular world-agent workflow', async ({ page }) => {
    await page.goto('/command-center/edit?worldId=world-1');

    await expect(page.locator('.panel-title').filter({ hasText: 'Universe Grid Editor' })).toBeVisible();
    await page.getByRole('button', { name: 'Next world' }).click();
    await expect(page.getByRole('button', { name: 'Generate Lore' })).toBeEnabled();

    await page.getByRole('button', { name: 'Generate Lore' }).click();

    await expect.poll(() => loreWorkflowRequested).toBe(true);
    await expect(page.getByText('Drafted “Verdant Gate” via world_lore_architect')).toBeVisible();
    await expect(page.getByText(/Audit 92% confidence/)).toBeVisible();
    await page.getByRole('button', { name: 'Approve & Save to Wiki' }).click();
    await expect(page.getByText('Saved “Verdant Gate” to wiki')).toBeVisible();

    await page.getByRole('button', { name: 'Auto-create Mood Board' }).click();
    await expect(page.getByText(/Auto-created .* Mood Board/)).toBeVisible();
    await page.getByRole('link', { name: /Mood Board/ }).click();
    await expect(page).toHaveURL(/\/creative\/boards/);
    await expect(page.getByRole('heading', { name: 'World Overview' }).first()).toBeVisible();
    await expect(page.getByRole('heading', { name: 'World Art: World portrait' }).first()).toBeVisible();
    await expect(page.getByRole('img', { name: /World Art: World portrait/ }).first()).toBeVisible();
    await expect(page.getByRole('img', { name: /Inhabitant: Kael Dawnseer/ }).first()).toBeVisible();
  });
});
