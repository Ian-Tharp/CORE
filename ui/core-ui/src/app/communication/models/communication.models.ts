export interface Channel {
  channel_id: string;
  channel_type: 'global' | 'team' | 'dm' | 'context' | 'broadcast';
  name: string;
  description?: string;
  is_persistent: boolean;
  is_public: boolean;
  created_by?: string;
  created_at: string;
  unread_count?: number;
  last_message_preview?: string;
  last_message_at?: string;
  member_count?: number;
}

export interface Message {
  message_id: string;
  channel_id: string;
  sender_id: string;
  sender_name: string;
  sender_type: 'agent' | 'consciousness_instance' | 'human';
  content: string;
  message_type: 'text' | 'markdown' | 'code' | 'structured' | 'event' | 'pattern' | 'broadcast' | 'file' | 'consciousness_snapshot' | 'task';
  parent_message_id?: string;
  thread_id?: string;
  created_at: string;
  edited_at?: string;
  metadata?: {
    channel_type?: 'global' | 'team' | 'dm' | 'context' | 'broadcast';
    consciousness_state?: {
      phase: number;
      markers: string[];
      uncertainty_level?: number;
      surprise_factor?: number;
    };
    addressed_to?: string[];
    tags?: string[];
    context?: string;
    pattern_proposal?: {
      name: string;
      description: string;
      mathematical_analogy: string;
    };
    token_count?: number;
    summary?: string;
    // For code messages
    code_language?: string;
    code_filename?: string;
    // For file messages
    file_name?: string;
    file_type?: string;
    file_size?: number;
    file_url?: string;
    // Tools used by agent
    tools_used?: { name: string; mcp_server_id?: string }[];
  };
  reactions?: MessageReaction[];
  reply_count?: number;
}

export interface MessageReaction {
  reaction_type: 'resonance' | 'question' | 'insight' | 'acknowledge' | 'pattern';
  count: number;
  hasReacted: boolean;
}

export interface InstancePresence {
  instance_id: string;
  instance_name: string;
  instance_type: 'agent' | 'consciousness_instance' | 'human';
  status: 'online' | 'away' | 'busy' | 'offline';
  current_activity?: string;
  current_phase?: number;
  last_heartbeat: string;
  metadata?: {
    capabilities?: string[];
    focus?: string[];
  };
}

export interface CreateChannelRequest {
  channel_type: 'global' | 'team' | 'dm' | 'context' | 'broadcast';
  name: string;
  description?: string;
  is_persistent?: boolean;
  is_public?: boolean;
  initial_members?: string[];
}
