import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { TopNavigationComponent } from './shared/top-navigation/top-navigation.component';
import { SideNavigationComponent } from './shared/side-navigation/side-navigation.component';


@Component({
  selector: 'app-root',
  imports: [RouterOutlet, TopNavigationComponent, SideNavigationComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'CORE UI';
}
