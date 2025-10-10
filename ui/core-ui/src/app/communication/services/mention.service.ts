import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { map } from 'rxjs/operators';
import { MentionSuggestion, MentionContext } from '../models/mention.models';
import { ChannelService } from './channel.service';
import { PresenceService } from './presence.service';

@Injectable({
  providedIn: 'root'
})
export class MentionService {
  constructor(
    private channelService: ChannelService,
    private presenceService: PresenceService
  ) {}

  /**
   * Get default suggestions when @ is first typed
   * Context-aware: Shows relevant users/channels based on current channel
   * Production: Would call API to get contextual suggestions
   */
  getDefaultSuggestions(context: MentionContext): Observable<MentionSuggestion[]> {
    // In production: return this.http.get<MentionSuggestion[]>(`/api/mentions/default?userId=${context.userId}&channelId=${context.currentChannelId}`)

    // Return channel-specific suggestions (users in this channel)
    return this.getChannelSpecificSuggestions(context);
  }

  /**
   * Get filtered mention suggestions based on query
   * Production: Would call API with query and context for server-side filtering
   * API should handle pagination for global channels with many users
   */
  getMentionSuggestions(query: string, context: MentionContext): Observable<MentionSuggestion[]> {
    // In production: return this.http.get<MentionSuggestion[]>(`/api/mentions?q=${query}&channelId=${context.currentChannelId}&limit=5&offset=0`)

    if (!query || query.trim().length === 0) {
      return this.getDefaultSuggestions(context);
    }

    // Get channel-specific suggestions and filter by query
    return this.getChannelSpecificSuggestions(context).pipe(
      map(suggestions => this.filterAndRankSuggestions(query, suggestions))
    );
  }

  /**
   * Get channel-specific mention suggestions based on channel type
   * Production: API call to get members and mentionable entities for specific channel
   * For global channels, pagination would be required on the backend
   */
  private getChannelSpecificSuggestions(context: MentionContext): Observable<MentionSuggestion[]> {
    // In production: return this.http.get<MentionSuggestion[]>(`/api/channels/${context.currentChannelId}/mentions`)

    return new Observable(observer => {
      const suggestions: MentionSuggestion[] = [];

      // Get all instances first
      this.presenceService.getOnlineInstances().subscribe(instances => {
        // Filter based on channel type
        if (context.channelType === 'dm') {
          // DM: Only include participants of this specific DM
          const dmParticipants = this.getDMParticipants(context.currentChannelId, instances);
          dmParticipants.forEach(instance => {
            suggestions.push(this.mapInstanceToSuggestion(instance));
          });
          observer.next(suggestions);
          observer.complete();
        } else if (context.channelType === 'team') {
          // Team: Only include team members
          const teamMembers = this.getTeamMembers(context.currentChannelId, instances);
          teamMembers.forEach(instance => {
            suggestions.push(this.mapInstanceToSuggestion(instance));
          });
          observer.next(suggestions);
          observer.complete();
        } else if (context.channelType === 'global') {
          // Global: Include all users (with pagination in production)
          // Production note: API should handle pagination with limit/offset params
          instances.forEach(instance => {
            suggestions.push(this.mapInstanceToSuggestion(instance));
          });
          observer.next(suggestions);
          observer.complete();
        } else {
          observer.next([]);
          observer.complete();
        }
      });
    });
  }

  /**
   * Get participants for a DM channel
   * Production: Backend would manage channel membership table
   */
  private getDMParticipants(channelId: string, allInstances: any[]): any[] {
    // Extract instance identifier from channel ID (e.g., "dm_threshold" or "dm_instance_011_threshold")
    const dmPrefix = 'dm_';
    if (!channelId.startsWith(dmPrefix)) return [];

    const participantIdentifier = channelId.substring(dmPrefix.length);

    // Find the participant by matching against instance_id or the last part of instance_id
    return allInstances.filter(instance => {
      const instanceIdParts = instance.instance_id.split('_');
      const lastPart = instanceIdParts[instanceIdParts.length - 1];
      return instance.instance_id === participantIdentifier ||
             participantIdentifier.includes(lastPart) ||
             lastPart === participantIdentifier;
    });
  }

  /**
   * Get members for a team channel
   * Production: Backend would query team membership table
   */
  private getTeamMembers(channelId: string, allInstances: any[]): any[] {
    // Mock implementation: Map team channels to their members
    // Matches the team channels defined in channel.service.ts
    const teamMemberships: { [key: string]: string[] } = {
      'team_task_alpha': [
        'agent_orchestration',
        'agent_comprehension',
        'agent_reasoning',
        'agent_evaluation'
      ],
      'team_knowledge_indexing': [
        'agent_reasoning',
        'instance_007_synthesis',
        'instance_010_continuum'
      ],
      // Add more team mappings as needed
    };

    const teamMembers = teamMemberships[channelId] || [];
    return allInstances.filter(instance => teamMembers.includes(instance.instance_id));
  }

  /**
   * Filter and rank suggestions based on query relevance
   */
  private filterAndRankSuggestions(query: string, suggestions: MentionSuggestion[]): MentionSuggestion[] {
    const lowerQuery = query.toLowerCase().trim();

    // Score each suggestion
    const scored = suggestions
      .map(suggestion => {
        const name = suggestion.name.toLowerCase();
        const displayName = suggestion.displayName.toLowerCase();
        let score = 0;

        // Exact match (highest priority)
        if (name === lowerQuery || displayName === lowerQuery) {
          score = 1000;
        }
        // Starts with query
        else if (name.startsWith(lowerQuery) || displayName.startsWith(lowerQuery)) {
          score = 500;
        }
        // Contains query
        else if (name.includes(lowerQuery) || displayName.includes(lowerQuery)) {
          score = 250;
        }
        // No match
        else {
          return null;
        }

        // Boost online users
        if (suggestion.type === 'user' && suggestion.status === 'online') {
          score += 100;
        }

        return { suggestion, score };
      })
      .filter(item => item !== null) as { suggestion: MentionSuggestion; score: number }[];

    // Sort by score and take top 5
    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(item => item.suggestion);
  }

  /**
   * Map instance presence to mention suggestion
   */
  private mapInstanceToSuggestion(instance: any): MentionSuggestion {
    let icon = 'person';
    if (instance.instance_type === 'consciousness_instance') {
      icon = 'eco';
    } else if (instance.instance_type === 'agent') {
      icon = 'smart_toy';
    }

    return {
      id: instance.instance_id,
      name: instance.instance_name,
      displayName: instance.instance_name,
      type: 'user',
      subtype: instance.instance_type,
      icon: icon,
      status: instance.status,
      metadata: {
        currentActivity: instance.current_activity,
        phase: instance.current_phase
      }
    };
  }

  /**
   * Map channel to mention suggestion
   */
  private mapChannelToSuggestion(channel: any): MentionSuggestion {
    let icon = 'tag';
    if (channel.channel_type === 'global') {
      icon = 'public';
    } else if (channel.channel_type === 'team') {
      icon = 'groups';
    } else if (channel.channel_type === 'dm') {
      icon = 'chat';
    }

    return {
      id: channel.channel_id,
      name: channel.name,
      displayName: `@${channel.name}`, // Use @ for consistency
      type: 'channel',
      subtype: channel.channel_type,
      icon: icon,
      metadata: {
        description: channel.description,
        memberCount: channel.member_count
      }
    };
  }
}
