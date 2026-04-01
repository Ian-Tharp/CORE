import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { TopNavigationComponent } from './shared/top-navigation/top-navigation.component';
import { SideNavigationComponent } from './shared/side-navigation/side-navigation.component';
import { DiscordBridgeService } from './services/discord-bridge/discord-bridge.service';


@Component({
  selector: 'app-root',
  imports: [RouterOutlet, TopNavigationComponent, SideNavigationComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit {
  title = 'CORE UI';

  constructor(private readonly discordBridgeService: DiscordBridgeService) {}

  public ngOnInit(): void {
    this.discordBridgeService.startStatusPolling();
  }
}
