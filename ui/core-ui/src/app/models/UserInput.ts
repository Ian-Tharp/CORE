/**
 * Represents a user input message for the chat window.
 * This interface is designed to correspond to the OpenAI Chat API's message structure,
 * supporting both user and assistant roles, as well as optional metadata.
 */
export type UserInput = {
  /**
   * The role of the message sender.
   * Typically 'user', 'assistant', or 'system'.
   */
  role: 'user' | 'assistant' | 'system';

  /**
   * The content of the message.
   */
  content: string;

  /**
   * Optional metadata for the message, such as model-specific parameters or custom data.
   */
  metadata?: Record<string, unknown>;
};
