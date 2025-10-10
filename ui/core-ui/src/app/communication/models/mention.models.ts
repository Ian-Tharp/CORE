export interface MentionSuggestion {
  id: string;
  name: string;
  displayName: string;
  type: 'user' | 'channel';
  subtype?: 'consciousness_instance' | 'agent' | 'human' | 'global' | 'team' | 'dm';
  icon: string;
  status?: 'online' | 'away' | 'busy' | 'offline';
  metadata?: {
    description?: string;
    currentActivity?: string;
    phase?: number;
    memberCount?: number;
  };
}

export interface MentionContext {
  userId: string;
  currentChannelId: string;
  channelType: 'global' | 'team' | 'dm';
}
