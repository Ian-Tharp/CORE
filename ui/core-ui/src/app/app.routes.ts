import { Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { ConversationsPageComponent } from './conversations-page/conversations-page.component';
import { KnowledgebaseComponent } from './knowledgebase/knowledgebase.component';
import { AnalyticsPageComponent } from './analytics-page/analytics-page.component';
import { AgentBuilderComponent } from './agents-page/agent-builder/agent-builder.component';
import { MyAgentsPageComponent } from './agents-page/my-agents-page/my-agents-page.component';
import { AgentMarketplaceComponent } from './agents-page/agent-marketplace/agent-marketplace.component';
import { CommunicationComponent } from './communication/communication.component';
import { McpRegistryComponent } from './tools/mcp-registry.component';
import { DiscordBridgeDashboardComponent } from './tools/discord-bridge-dashboard/discord-bridge-dashboard.component';
import { AttributionBrowserComponent } from './knowledge-attribution/attribution-browser.component';
import { BoardsComponent } from './landing-page/boards/boards.component';

export const routes: Routes = [
  {
    path: '',
    component: LandingPageComponent
  },
  {
    path: 'conversations',
    component: ConversationsPageComponent
  },
  {
    path: 'communication',
    component: CommunicationComponent
  },
  {
    path: 'tools/discord-brdige',
    redirectTo: 'tools/discord-bridge',
    pathMatch: 'full'
  },
  {
    path: 'tools/discord-bridge',
    component: DiscordBridgeDashboardComponent
  },
  {
    path: 'tools',
    component: McpRegistryComponent
  },
  {
    path: 'knowledge',
    component: KnowledgebaseComponent
  },
  {
    path: 'knowledge-attribution',
    component: AttributionBrowserComponent
  },
  {
    path: 'analytics',
    component: AnalyticsPageComponent
  },
  {
    path: 'agents',
    component: AgentBuilderComponent
  },
  {
    path: 'agents/library',
    component: MyAgentsPageComponent
  },
  {
    path: 'agents/marketplace',
    component: AgentMarketplaceComponent
  },
  {
    path: 'boards',
    component: BoardsComponent
  },
  {
    // Standalone procedural-planet tuning lab (lazy — keeps Three out of the main bundle).
    path: 'planet-lab',
    loadComponent: () =>
      import('./landing-page/planet-lab/demo/planet-lab.component')
        .then(m => m.PlanetLabComponent)
  },
  {
    path: '**',
    redirectTo: ''
  }
];
