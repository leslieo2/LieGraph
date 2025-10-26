/**
 * LangGraph Server Client
 * Direct connection to LangGraph Server without custom backend
 */

import { Client } from '@langchain/langgraph-sdk';

class LangGraphClient {
  constructor() {
    const apiUrl = process.env.REACT_APP_LANGGRAPH_URL || 'http://localhost:8124';
    this.client = new Client({ apiUrl });
    this.apiUrl = apiUrl;
    this.threadId = null;
    this.assistantId = 'who_is_spy';
  }

  /**
   * Create new game thread
   */
  async createGameThread() {
    try {
      const thread = await this.client.threads.create();
      this.threadId = thread.thread_id;
      console.log('Created game thread:', this.threadId);
      return this.threadId;
    } catch (error) {
      console.error('Failed to create game thread:', error);
      throw error;
    }
  }

  /**
   * Start game workflow
   */
  async startGame(civilian_word, spy_word) {
    if (!this.threadId) {
      await this.createGameThread();
    }

    const input = {
      game_id: `game-${Date.now()}`,
    };

    if (civilian_word && spy_word) {
      input.host_private_state = {
        civilian_word: civilian_word,
        spy_word: spy_word
      };
    }

    // The SDK's stream method returns an async iterator that we can pass directly to the UI
    return this.client.runs.stream(
      this.threadId,
      this.assistantId,
      {
        input: input,
        stream_mode: ['values', 'updates']
      }
    );
  }

  /**
   * Convert LangGraph state to UI format
   * Directly use backend state without additional calculations
   */
  convertToUIFormat(state, stream = false) {
    if (!state) {
      return this.getDefaultState();
    }

    const baseState = {
      status: state.game_phase || 'setup',
      current_round: state.current_round || 1,
      players: state.players || [],
      completed_speeches: state.completed_speeches || [],
      current_votes: state.current_votes || {},
      winner: state.winner,
      eliminated_players: state.eliminated_players || [],
      player_private_states: state.player_private_states || {},
      host_private_state: state.host_private_state || {},
      raw_state: state
    };

    if (stream) {
      // In streaming mode, we need to merge state rather than replace
      return (prevState) => ({
        ...prevState,
        ...baseState,
        // Deep merge private states to ensure no data loss
        player_private_states: {
          ...prevState.player_private_states,
          ...Object.keys(baseState.player_private_states).reduce((acc, playerId) => ({
            ...acc,
            [playerId]: {
              ...prevState.player_private_states?.[playerId],
              ...baseState.player_private_states[playerId]
            }
          }), {})
        },
        host_private_state: {
          ...prevState.host_private_state,
          ...baseState.host_private_state
        }
      });
    } else {
      // In non-streaming mode, directly return new state
      return baseState;
    }
  }



  /**
   * Get default state
   */
  getDefaultState() {
    return {
      status: 'loading',
      current_round: 1,
      players: [],
      completed_speeches: [],
      current_votes: {},
      winner: null,
      eliminated_players: [],
      player_private_states: {},
      host_private_state: {}
    };
  }


  /**
   * Reset game
   */
  async resetGame() {
    this.threadId = null;
    await this.createGameThread();
  }
}

export default LangGraphClient;
