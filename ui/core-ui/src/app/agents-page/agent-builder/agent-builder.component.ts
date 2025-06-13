import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatChipsModule } from '@angular/material/chips';
import { MatRadioModule } from '@angular/material/radio';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatTabsModule } from '@angular/material/tabs';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { trigger, transition, style, animate } from '@angular/animations';

interface Step {
  id: string;
  label: string;
  icon: string;
}

interface AgentType {
  id: string;
  name: string;
  icon: string;
  description: string;
}

interface PromptTemplate {
  id: string;
  name: string;
  prompt: string;
}

interface Document {
  id: string;
  name: string;
  size: string;
  type: string;
}

interface McpServer {
  id: string;
  name: string;
  description: string;
  icon: string;
  toolCount: number;
  category: string;
  connected: boolean;
}

interface Tool {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
}

interface ToolCategory {
  id: string;
  name: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
}

interface TestScenario {
  id: string;
  title: string;
  description: string;
  prompt: string;
}

interface AIModel {
  id: string;
  name: string;
  provider: string;
  description: string;
}

@Component({
  selector: 'app-agent-builder',
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatChipsModule,
    MatRadioModule,
    MatCheckboxModule,
    MatTabsModule,
    MatExpansionModule,
    MatProgressSpinnerModule,
    MatTooltipModule
  ],
  templateUrl: './agent-builder.component.html',
  styleUrl: './agent-builder.component.scss',
  animations: [
    trigger('slideIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateX(50px)' }),
        animate('400ms ease-out', style({ opacity: 1, transform: 'translateX(0)' }))
      ])
    ])
  ]
})
export class AgentBuilderComponent implements OnInit {
  currentStepIndex = 0;
  progressPercentage = 0;
  isDragging = false;
  selectedTemplate: string | null = null;
  mcpSearchQuery = '';
  testPrompt = '';
  isTestLoading = false;
  chatMessages: ChatMessage[] = [];

  steps: Step[] = [
    { id: 'core', label: 'Core Config', icon: 'settings' },
    { id: 'knowledge', label: 'Knowledge Base', icon: 'school' },
    { id: 'tools', label: 'Tools & Capabilities', icon: 'build' },
    { id: 'test', label: 'Test & Validate', icon: 'science' },
    { id: 'review', label: 'Review & Deploy', icon: 'rocket_launch' }
  ];

  agentTypes: AgentType[] = [
    { id: 'general', name: 'General Purpose', icon: 'psychology', description: 'Versatile agent for various tasks' },
    { id: 'analyst', name: 'Data Analyst', icon: 'analytics', description: 'Specialized in data analysis and insights' },
    { id: 'coder', name: 'Code Assistant', icon: 'code', description: 'Expert in programming and debugging' },
    { id: 'researcher', name: 'Research Agent', icon: 'science', description: 'Focused on information gathering' },
    { id: 'automation', name: 'Automation Agent', icon: 'precision_manufacturing', description: 'Task automation specialist' }
  ];

  promptTemplates: PromptTemplate[] = [
    { id: 'blank', name: 'Blank Template', prompt: '' },
    { id: 'helpful', name: 'Helpful Assistant', prompt: 'You are a helpful AI assistant...' },
    { id: 'technical', name: 'Technical Expert', prompt: 'You are a technical expert with deep knowledge...' },
    { id: 'creative', name: 'Creative Partner', prompt: 'You are a creative thinking partner...' }
  ];

  aiModels: AIModel[] = [
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'OpenAI', description: 'Most capable model with 128k context' },
    { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', description: 'Advanced reasoning and generation' },
    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'OpenAI', description: 'Fast and cost-effective' },
    { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', description: 'Top-tier reasoning and analysis' },
    { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', description: 'Balanced performance' },
    { id: 'claude-3-haiku', name: 'Claude 3 Haiku', provider: 'Anthropic', description: 'Fast and efficient' },
    { id: 'gemini-pro', name: 'Gemini Pro', provider: 'Google', description: 'Multimodal capabilities' },
    { id: 'llama-3-70b', name: 'Llama 3 70B', provider: 'Meta', description: 'Open source powerhouse' }
  ];

  testScenarios: TestScenario[] = [
    {
      id: 'greeting',
      title: 'Basic Greeting',
      description: 'Test how your agent introduces itself',
      prompt: 'Hello! Can you introduce yourself and explain what you can help me with?'
    },
    {
      id: 'task-handling',
      title: 'Task Handling',
      description: 'Test how your agent handles specific tasks',
      prompt: 'I need help analyzing some data. What\'s your approach to data analysis?'
    },
    {
      id: 'edge-case',
      title: 'Edge Case Handling',
      description: 'Test how your agent handles unclear requests',
      prompt: 'Can you do the thing with the stuff?'
    },
    {
      id: 'tool-usage',
      title: 'Tool Usage',
      description: 'Test how your agent uses available tools',
      prompt: 'Can you show me what tools you have access to and how you would use them?'
    }
  ];

  testMetrics = {
    responseTime: 0,
    tokenCount: 0,
    toolCalls: 0,
    contextUsage: 0
  };

  uploadedDocuments: Document[] = [];

  mcpServers: McpServer[] = [
    {
      id: 'home-assistant',
      name: 'Home Assistant MCP',
      description: 'Control smart home devices and automation',
      icon: 'home',
      toolCount: 45,
      category: 'Smart Home',
      connected: false
    },
    {
      id: 'github',
      name: 'GitHub MCP',
      description: 'Interact with GitHub repositories and issues',
      icon: 'github',
      toolCount: 32,
      category: 'Development',
      connected: false
    },
    {
      id: 'filesystem',
      name: 'FileSystem MCP',
      description: 'Read and write files on the local system',
      icon: 'folder',
      toolCount: 12,
      category: 'System',
      connected: false
    },
    {
      id: 'web-browser',
      name: 'Web Browser MCP',
      description: 'Browse and extract information from websites',
      icon: 'public',
      toolCount: 8,
      category: 'Web',
      connected: false
    }
  ];

  toolCategories: ToolCategory[] = [
    { id: 'data', name: 'Data Processing' },
    { id: 'communication', name: 'Communication' },
    { id: 'system', name: 'System Control' },
    { id: 'analysis', name: 'Analysis' }
  ];

  availableTools: Tool[] = [
    { id: 'file-reader', name: 'File Reader', description: 'Read and parse various file formats', icon: 'description', category: 'data' },
    { id: 'data-transform', name: 'Data Transformer', description: 'Transform and manipulate data', icon: 'transform', category: 'data' },
    { id: 'email-sender', name: 'Email Sender', description: 'Send emails programmatically', icon: 'email', category: 'communication' },
    { id: 'web-scraper', name: 'Web Scraper', description: 'Extract data from websites', icon: 'web', category: 'data' },
    { id: 'system-monitor', name: 'System Monitor', description: 'Monitor system resources', icon: 'monitor_heart', category: 'system' },
    { id: 'sentiment-analyzer', name: 'Sentiment Analyzer', description: 'Analyze text sentiment', icon: 'sentiment_satisfied', category: 'analysis' }
  ];

  selectedTools: Tool[] = [];

  agentConfig = {
    name: '',
    description: '',
    type: '',
    model: '',
    systemPrompt: '',
    embeddingModel: '',
    chunkSize: 500,
    chunkOverlap: 50,
    retrievalStrategy: 'similarity'
  };

  deploymentConfig = {
    resources: 'standard',
    autoStart: false,
    enableLogging: true,
    exposeApi: false
  };

  get filteredMcpServers(): McpServer[] {
    if (!this.mcpSearchQuery) return this.mcpServers;
    const query = this.mcpSearchQuery.toLowerCase();
    return this.mcpServers.filter(server => 
      server.name.toLowerCase().includes(query) ||
      server.description.toLowerCase().includes(query) ||
      server.category.toLowerCase().includes(query)
    );
  }

  ngOnInit(): void {
    this.updateProgress();
  }

  updateProgress(): void {
    this.progressPercentage = ((this.currentStepIndex + 1) / this.steps.length) * 100;
  }

  goToStep(index: number): void {
    if (index <= this.currentStepIndex || this.canProceed()) {
      this.currentStepIndex = index;
      this.updateProgress();
    }
  }

  nextStep(): void {
    if (this.currentStepIndex < this.steps.length - 1 && this.canProceed()) {
      this.currentStepIndex++;
      this.updateProgress();
    } else if (this.currentStepIndex === this.steps.length - 1) {
      this.deployAgent();
    }
  }

  previousStep(): void {
    if (this.currentStepIndex > 0) {
      this.currentStepIndex--;
      this.updateProgress();
    }
  }

  canProceed(): boolean {
    switch (this.currentStepIndex) {
      case 0:
        return !!this.agentConfig.name && !!this.agentConfig.type && !!this.agentConfig.systemPrompt && !!this.agentConfig.model;
      case 1:
        return true; // Knowledge base is optional
      case 2:
        return true; // Tools are optional
      case 3:
        return true; // Testing is optional but recommended
      case 4:
        return true; // Review step
      default:
        return false;
    }
  }

  selectAgentType(typeId: string): void {
    this.agentConfig.type = typeId;
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging = false;
    // Placeholder for file handling
    console.log('Files dropped - implementation pending');
  }

  getFileIcon(type: string): string {
    const iconMap: { [key: string]: string } = {
      'pdf': 'picture_as_pdf',
      'txt': 'description',
      'md': 'description',
      'docx': 'description',
      'csv': 'table_chart'
    };
    return iconMap[type] || 'insert_drive_file';
  }

  removeDocument(docId: string): void {
    this.uploadedDocuments = this.uploadedDocuments.filter(doc => doc.id !== docId);
  }

  toggleMcpServer(server: McpServer): void {
    server.connected = !server.connected;
  }

  getToolsByCategory(categoryId: string): Tool[] {
    return this.availableTools.filter(tool => tool.category === categoryId);
  }

  isToolSelected(toolId: string): boolean {
    return this.selectedTools.some(tool => tool.id === toolId);
  }

  toggleTool(tool: Tool): void {
    const index = this.selectedTools.findIndex(t => t.id === tool.id);
    if (index >= 0) {
      this.selectedTools.splice(index, 1);
    } else {
      this.selectedTools.push(tool);
    }
  }

  removeTool(tool: Tool): void {
    this.selectedTools = this.selectedTools.filter(t => t.id !== tool.id);
  }

  deployAgent(): void {
    console.log('Deploying agent with config:', {
      agentConfig: this.agentConfig,
      documents: this.uploadedDocuments,
      tools: this.selectedTools,
      deploymentConfig: this.deploymentConfig
    });
    // Placeholder for deployment logic
  }

  saveAsTemplate(): void {
    console.log('Saving as template...');
    // Placeholder for template saving logic
  }

  async sendTestMessage(): Promise<void> {
    if (!this.testPrompt.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: this.testPrompt,
      timestamp: new Date()
    };

    this.chatMessages.push(userMessage);
    this.testPrompt = '';
    this.isTestLoading = true;

    // Simulate agent response
    setTimeout(() => {
      const agentMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: this.generateTestResponse(userMessage.content),
        timestamp: new Date()
      };
      this.chatMessages.push(agentMessage);
      this.isTestLoading = false;
      this.updateTestMetrics();
    }, 1500);
  }

  generateTestResponse(prompt: string): string {
    // Simulate different responses based on the agent configuration
    const agentName = this.agentConfig.name || 'Assistant';
    const agentType = this.agentConfig.type;

    if (prompt.toLowerCase().includes('introduce')) {
      return `Hello! I'm ${agentName}, your ${agentType} agent. ${this.agentConfig.description || 'I\'m here to help you with various tasks.'}`;
    } else if (prompt.toLowerCase().includes('tools')) {
      const toolList = this.selectedTools.map(t => t.name).join(', ');
      return `I have access to the following tools: ${toolList || 'No tools selected yet'}. I can use these to help you accomplish various tasks.`;
    } else {
      return `As ${agentName}, I understand you're asking about "${prompt}". Based on my configuration as a ${agentType} agent, I would approach this by leveraging my knowledge base and available tools to provide you with the best possible assistance.`;
    }
  }

  loadTestScenario(scenario: TestScenario): void {
    this.testPrompt = scenario.prompt;
  }

  clearTestChat(): void {
    this.chatMessages = [];
    this.testMetrics = {
      responseTime: 0,
      tokenCount: 0,
      toolCalls: 0,
      contextUsage: 0
    };
  }

  updateTestMetrics(): void {
    // Simulate metrics update
    this.testMetrics.responseTime = Math.floor(Math.random() * 2000) + 500;
    this.testMetrics.tokenCount = Math.floor(Math.random() * 500) + 100;
    this.testMetrics.toolCalls = Math.floor(Math.random() * 3);
    this.testMetrics.contextUsage = Math.floor(Math.random() * 50) + 20;
  }

  handleKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendTestMessage();
    }
  }
}
