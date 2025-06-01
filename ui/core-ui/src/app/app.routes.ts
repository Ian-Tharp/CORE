import { Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { ConversationsPageComponent } from './conversations-page/conversations-page.component';

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
    path: '**',
    redirectTo: ''
  }
];
