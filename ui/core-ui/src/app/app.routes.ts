import { Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { ConversationsPageComponent } from './conversations-page/conversations-page.component';
import { KnowledgebaseComponent } from './knowledgebase/knowledgebase.component';
import { AnalyticsPageComponent } from './analytics-page/analytics-page.component';

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
    path: '**',
    redirectTo: ''
  }
];
