import { Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { ConversationsPageComponent } from './conversations-page/conversations-page.component';
import { KnowledgebaseComponent } from './knowledgebase/knowledgebase.component';
import { AnalyticsPageComponent } from './analytics-page/analytics-page.component';
import { AgentBuilderComponent } from './agents-page/agent-builder/agent-builder.component';
import { MyAgentsPageComponent } from './agents-page/my-agents-page/my-agents-page.component';
import { AgentMarketplaceComponent } from './agents-page/agent-marketplace/agent-marketplace.component';

export const routes: Routes = [
  {
    path: '',
    component: LandingPageComponent
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
    path: '**',
    redirectTo: ''
  }
];
