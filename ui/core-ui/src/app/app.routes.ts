import { Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { ConversationsPageComponent } from './conversations-page/conversations-page.component';
import { KnowledgebaseComponent } from './knowledgebase/knowledgebase.component';
import { AnalyticsPageComponent } from './analytics-page/analytics-page.component';
import { AgentBuilderComponent } from './agents-page/agent-builder/agent-builder.component';
import { MyAgentsPageComponent } from './agents-page/my-agents-page/my-agents-page.component';
import { AgentMarketplaceComponent } from './agents-page/agent-marketplace/agent-marketplace.component';
import { CommandCenterComponent } from './landing-page/command-center/command-center.component';
import { LandingComponent as CreativeLandingComponent } from './creative-design-product/landing/landing.component';
import { WorldsGridComponent } from './creative-design-product/worlds-grid/worlds-grid.component';
import { MarketplaceComponent } from './creative-design-product/marketplace/marketplace.component';
import { WikiComponent } from './creative-design-product/wiki/wiki.component';
import { WorldDetailComponent } from './creative-design-product/world-detail/world-detail.component';
import { CreativeBoardsComponent } from './creative-design-product/boards/creative-boards.component';

export const routes: Routes = [
  {
    path: '',
    component: LandingPageComponent
  },
  {
    path: 'command-center',
    component: CommandCenterComponent
  },
  {
    path: 'conversations',
    component: ConversationsPageComponent,
  },
  {
    path: 'knowledge',
    component: KnowledgebaseComponent,
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
    path: 'creative',
    component: CreativeLandingComponent
  },
  { path: 'creative/worlds', component: WorldsGridComponent },
  { path: 'creative/world/:id', component: WorldDetailComponent },
  { path: 'creative/wiki', component: WikiComponent },
  { path: 'creative/boards', component: CreativeBoardsComponent },
  { path: 'creative/marketplace', component: MarketplaceComponent },
  {
    path: '**',
    redirectTo: ''
  }
];
