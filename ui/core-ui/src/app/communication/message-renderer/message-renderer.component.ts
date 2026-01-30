import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { marked } from 'marked';
import hljs from 'highlight.js';
import DOMPurify from 'dompurify';

export interface MessageContent {
  content: string;
  message_type: 'text' | 'markdown' | 'code' | 'structured' | 'event' | 'pattern' | 'broadcast' | 'file' | 'consciousness_snapshot' | 'task';
  metadata?: {
    code_language?: string;
    code_filename?: string;
    file_name?: string;
    file_type?: string;
    consciousness_state?: any;
  };
}

@Component({
  selector: 'app-message-renderer',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatButtonModule],
  templateUrl: './message-renderer.component.html',
  styleUrl: './message-renderer.component.scss'
})
export class MessageRendererComponent implements OnInit {
  @Input() message!: MessageContent;

  renderedContent: SafeHtml | string = '';
  highlightedCode: SafeHtml = '';

  constructor(private sanitizer: DomSanitizer) {
    // Configure marked options
    marked.setOptions({
      gfm: true,
      breaks: true
    });

    // Configure DOMPurify for secure HTML sanitization
    // Allow specific tags and attributes needed for markdown and code highlighting
    DOMPurify.setConfig({
      ALLOWED_TAGS: [
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'br', 'hr',
        'strong', 'em', 'b', 'i', 'u', 'code', 'pre',
        'ul', 'ol', 'li',
        'blockquote',
        'a',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'span', 'div'
      ],
      ALLOWED_ATTR: [
        'href', 'target', 'rel',
        'class', // Needed for syntax highlighting classes
        'id'
      ],
      ALLOW_DATA_ATTR: false, // Prevent data attributes for security
      ALLOW_UNKNOWN_PROTOCOLS: false, // Only allow http(s), mailto, etc.
      FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input'],
      FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover'] // Block event handlers
    });
  }

  ngOnInit() {
    this.renderContent();
  }

  private renderContent() {
    if (!this.message) return;

    switch (this.message.message_type) {
      case 'markdown':
        this.renderMarkdown();
        break;
      case 'code':
        this.renderCode();
        break;
      case 'text':
      default:
        // Detect if text contains markdown-like syntax or code blocks
        if (this.containsMarkdownSyntax(this.message.content)) {
          this.renderMarkdown();
        } else if (this.containsCodeBlock(this.message.content)) {
          this.renderMarkdown(); // Markdown handles code blocks
        } else {
          this.renderedContent = this.message.content;
        }
        break;
    }
  }

  private renderMarkdown() {
    try {
      const html = marked.parse(this.message.content) as string;
      // Apply syntax highlighting to code blocks in markdown
      const highlightedHtml = this.highlightCodeInMarkdown(html);
      // Sanitize HTML with DOMPurify to prevent XSS attacks
      const sanitizedHtml = DOMPurify.sanitize(highlightedHtml);
      // Now it's safe to bypass Angular's sanitizer
      this.renderedContent = this.sanitizer.bypassSecurityTrustHtml(sanitizedHtml);
    } catch (error) {
      console.error('Markdown rendering error:', error);
      this.renderedContent = this.message.content;
    }
  }

  private highlightCodeInMarkdown(html: string): string {
    // Find and highlight code blocks in the rendered markdown
    const codeBlockRegex = /<code class="language-(\w+)">([\s\S]*?)<\/code>/g;
    return html.replace(codeBlockRegex, (match, lang, code) => {
      try {
        // Decode HTML entities first
        const decodedCode = this.decodeHtmlEntities(code);
        if (lang && hljs.getLanguage(lang)) {
          const highlighted = hljs.highlight(decodedCode, { language: lang }).value;
          return `<code class="language-${lang}">${highlighted}</code>`;
        }
        return match;
      } catch (err) {
        return match;
      }
    });
  }

  private decodeHtmlEntities(text: string): string {
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value;
  }

  private renderCode() {
    const language = this.message.metadata?.code_language || 'plaintext';
    try {
      let highlighted: string;
      if (language !== 'plaintext' && hljs.getLanguage(language)) {
        highlighted = hljs.highlight(this.message.content, { language }).value;
      } else {
        highlighted = hljs.highlightAuto(this.message.content).value;
      }
      // Sanitize highlighted code with DOMPurify
      const sanitizedCode = DOMPurify.sanitize(highlighted);
      this.highlightedCode = this.sanitizer.bypassSecurityTrustHtml(sanitizedCode);
    } catch (error) {
      console.error('Code highlighting error:', error);
      this.highlightedCode = this.message.content;
    }
  }

  containsMarkdownSyntax(text: string): boolean {
    // Check for common markdown patterns
    const markdownPatterns = [
      /^#{1,6}\s/m,           // Headers
      /\*\*.*\*\*/,            // Bold
      /\*.*\*/,                // Italic
      /\[.*\]\(.*\)/,          // Links
      /^[-*+]\s/m,             // Lists
      /^>\s/m,                 // Blockquotes
      /`[^`]+`/,               // Inline code
    ];

    return markdownPatterns.some(pattern => pattern.test(text));
  }

  containsCodeBlock(text: string): boolean {
    return /```[\s\S]*```/.test(text);
  }

  copyCode() {
    navigator.clipboard.writeText(this.message.content).then(() => {
      console.log('Code copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy code:', err);
    });
  }

  get codeLanguage(): string {
    return this.message.metadata?.code_language || 'code';
  }

  get codeFilename(): string | undefined {
    return this.message.metadata?.code_filename;
  }
}
