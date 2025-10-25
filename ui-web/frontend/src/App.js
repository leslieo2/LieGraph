import React, { useEffect, useState, useRef } from 'react';
import './App.css';
import LangGraphClient from './LangGraphClient';
import { useTranslation } from 'react-i18next';

function App() {
  const { t, i18n } = useTranslation();
  const [gameState, setGameState] = useState({
    status: 'loading',
    players: [],
    completed_speeches: [],
    current_votes: {},
    winner: null,
    eliminated_players: [],
    player_private_states: {},
    host_private_state: {},
    current_round: 1
  });
  const [isGameRunning, setIsGameRunning] = useState(false);
  const [civilianWord, setCivilianWord] = useState('');
  const [spyWord, setSpyWord] = useState('');
  const numPlayers = gameState.players && gameState.players.length > 0 ? gameState.players.length : 6;

  const [expandedSuspicionReasons, setExpandedSuspicionReasons] = useState({});
  const langGraphClient = useRef(null);
  const conversationContainerRef = useRef(null);

  const toggleSuspicionReason = (suspicionId) => {
    setExpandedSuspicionReasons(prev => ({
      ...prev,
      [suspicionId]: !prev[suspicionId]
    }));
  };

  useEffect(() => {
    langGraphClient.current = new LangGraphClient();
    return () => {
      if (langGraphClient.current) {
        // Cleanup logic can be added here
      }
    };
  }, []);

  useEffect(() => {
    if (conversationContainerRef.current) {
      conversationContainerRef.current.scrollTop = conversationContainerRef.current.scrollHeight;
    }
  }, [gameState.completed_speeches]);

  const startGame = async () => {
    try {
      setIsGameRunning(true);
      const stream = await langGraphClient.current.startGame(civilianWord, spyWord);
      const processStream = async () => {
        try {
          for await (const chunk of stream) {
            if (chunk.data) {
              const uiState = langGraphClient.current.convertToUIFormat(chunk.data, true);
              if (typeof uiState === 'function') {
                setGameState(uiState);
              } else {
                setGameState(prevState => ({ ...prevState, ...uiState }));
              }
            }
          }
        } catch (streamError) {
          console.error('Stream processing error:', streamError);
        }
      };
      processStream();
    } catch (error) {
      console.error('Failed to start game:', error);
      setIsGameRunning(false);
      setGameState({
        status: 'error',
        players: [],
        completed_speeches: [],
        current_votes: {},
        winner: null,
        eliminated_players: [],
        player_private_states: {},
        host_private_state: {},
        current_round: 1
      });
    }
  };

  const restartGame = async () => {
    try {
      // Reset game state to initial loading state
      setGameState({
        status: 'loading',
        players: [],
        completed_speeches: [],
        current_votes: {},
        winner: null,
        eliminated_players: [],
        player_private_states: {},
        host_private_state: {},
        current_round: 1
      });

      // Stop the current game and reset backend
      setIsGameRunning(false);
      await langGraphClient.current.resetGame();

      // Keep the custom words for the next game
      // The user can now see the start screen with their custom words preserved
    } catch (error) {
      console.error('Failed to restart game:', error);
      setIsGameRunning(false);
    }
  };

  const getPhaseDisplayName = (phase) => {
    const phaseMap = {
      'setup': t('phase_setup'),
      'speaking': t('phase_speaking'),
      'voting': t('phase_voting'),
      'result': t('phase_result')
    };
    return phaseMap[phase] || phase;
  };

  const getPhaseColor = (phase) => {
    const colorMap = {
      'setup': '#95a5a6',
      'speaking': '#38a169',
      'voting': '#e74c3c',
      'result': '#27ae60'
    };
    return colorMap[phase] || '#95a5a6';
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="game-header">
          <h1>{t('title')}</h1>
          {isGameRunning ? (
            <div className="game-status">
              <div className="phase-indicator" style={{ backgroundColor: getPhaseColor(gameState.status) }}>
                {getPhaseDisplayName(gameState.status)}
              </div>
              <div className="round-indicator">
                {t('round_indicator', { round: gameState.current_round || 1 })}
              </div>
              <div className="game-controls">
                <button className="control-button language-button" onClick={() => changeLanguage('en')}>{t('language_english')}</button>
                <button className="control-button language-button" onClick={() => changeLanguage('zh')}>{t('language_chinese')}</button>
                <button
                  className="control-button start-button"
                  onClick={startGame}
                  disabled={isGameRunning}
                >
                  {isGameRunning ? t('game_in_progress') : t('start_game')}
                </button>
                <button
                  className="control-button restart-button"
                  onClick={restartGame}
                  disabled={!isGameRunning}
                >
                  {t('restart_game')}
                </button>
              </div>
            </div>
          ) : (
            <div className="game-controls">
              <button className="control-button language-button" onClick={() => changeLanguage('en')}>{t('language_english')}</button>
              <button className="control-button language-button" onClick={() => changeLanguage('zh')}>{t('language_chinese')}</button>
            </div>
          )}
        </div>

        {!isGameRunning && gameState.status === 'loading' && (
          <div className="welcome-screen">
            <div className="welcome-content">
              <h2>{t('welcome_title')}</h2>
              <p>{t('welcome_subtitle')}</p>
              <div className="game-settings">
                <div>
                  <label htmlFor="civilian-word">{t('civilian_word_label')}</label>
                  <input id="civilian-word" type="text" value={civilianWord} onChange={(e) => setCivilianWord(e.target.value)} placeholder={t('civilian_word_placeholder')} />
                </div>
                <div>
                  <label htmlFor="spy-word">{t('spy_word_label')}</label>
                  <input id="spy-word" type="text" value={spyWord} onChange={(e) => setSpyWord(e.target.value)} placeholder={t('spy_word_placeholder')} />
                </div>
                <div className="player-count-info">
                  <strong>{t('player_count_info')}</strong> {t('player_count_unit', { numPlayers })}
                </div>
              </div>
              <p>{t('welcome_instruction')}</p>
              <button
                className="welcome-start-button"
                onClick={startGame}
              >
                {t('start_game')}
              </button>
            </div>
          </div>
        )}

        <div className="game-layout" style={{ display: isGameRunning ? 'flex' : 'none' }}>
          <div className="players-panel">
            <h3>{t('players_list_title')}</h3>
            <div className="players-list">
              {gameState.players.map(playerId => {
                const isEliminated = gameState.eliminated_players.includes(playerId);
                const privateState = gameState.player_private_states?.[playerId] || {};
                const playerRole = gameState.host_private_state?.player_roles?.[playerId];
                const civilianWord = gameState.host_private_state?.civilian_word;
                const spyWord = gameState.host_private_state?.spy_word;
                const assignedWord = privateState.assigned_word ||
                                   (playerRole === 'civilian' ? civilianWord :
                                    playerRole === 'spy' ? spyWord : t('role_unknown'));
                const selfBelief = privateState.playerMindset?.self_belief;

                return (
                  <div
                    key={playerId}
                    className={`player-item ${isEliminated ? 'eliminated' : ''}`}
                  >
                    <div className="player-avatar">
                      {isEliminated ? '💀' :
                       selfBelief?.role === 'civilian' ? '👤' : '🕵️'}
                    </div>
                    <div className="player-details">
                      <div className="player-name">{playerId}</div>
                      <div className="player-word">
                        <strong>{t('word_label')}</strong> {assignedWord}
                      </div>
                      <div className="player-belief">
                        <strong>{t('self_belief_label')}</strong>
                        {selfBelief && selfBelief.role && selfBelief.confidence !== undefined ? (
                          <div className="confidence-bar-container">
                            <span className={`belief-${selfBelief.role}`}>
                              {selfBelief.role === 'civilian' ? t('role_civilian') : t('role_spy')}
                            </span>
                            <div className="confidence-bar">
                              <div
                                className="confidence-fill"
                                style={{ width: `${(selfBelief.confidence * 100).toFixed(0)}%` }}
                              ></div>
                            </div>
                            <span className="confidence-percentage">
                              ({(selfBelief.confidence * 100).toFixed(0)}%)
                            </span>
                          </div>
                        ) : t('role_unknown')}
                      </div>
                      <div className="player-status">
                        {isEliminated ? t('status_eliminated') : t('status_in_game')}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="conversation-panel">
            <h3>{t('conversation_history_title')}</h3>
            <div className="conversation-container" ref={conversationContainerRef}>
              {gameState.completed_speeches && gameState.completed_speeches.length > 0 ? (
                <div className="speech-history">
                  {gameState.completed_speeches.map((speech, index) => (
                    <div key={index} className="speech-entry">
                      <div className="speech-header">
                        <span className="speaker-name">{speech.player_id}</span>
                        <span className="speech-round">{t('speech_round', { round: speech.round })}</span>
                        <span className="speech-time">
                          {new Date(speech.ts).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="speech-content">
                        {speech.content}
                      </div>
                      {speech.reasoning && (
                        <div className="reasoning-section">
                          <div className="reasoning-label">{t('reasoning_label')}</div>
                          <div className="reasoning-content">{speech.reasoning}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-conversation">
                  <p>{t('no_conversation_history')}</p>
                  <p>{t('game_starting_soon')}</p>
                </div>
              )}

              {gameState.status === 'voting' && gameState.current_votes && (
                <div className="voting-results">
                  <h4>{t('voting_results_title')}</h4>
                  <div className="votes-grid">
                    {Object.entries(gameState.current_votes).map(([voter, vote]) => (
                      <div key={voter} className="vote-item">
                        <span className="voter">{voter}</span>
                        <span className="vote-arrow">→</span>
                        <span className="vote-target">{vote.target}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {gameState.status === 'result' && gameState.winner && (
                <div className="game-result">
                  <h4 className={`result-${gameState.winner}`}>
                    {gameState.winner === 'civilians' ? t('civilians_win') : t('spy_wins')}
                  </h4>
                  <div className="result-details">
                    <p>{t('game_over')}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="suspicion-matrix-panel">
            <h3>{t('suspicion_matrix_title')}</h3>
            <div className="suspicion-matrix-content">
              {gameState.players.map(observerId => (
                <div key={observerId} className="observer-suspicions">
                  <h4>{t('observer_suspicion_title', { observerId })}</h4>
                  {gameState.player_private_states?.[observerId]?.playerMindset?.suspicions && Object.keys(gameState.player_private_states[observerId].playerMindset.suspicions).length > 0 ? (
                    <div className="suspicions-list">
                      {Object.entries(gameState.player_private_states[observerId].playerMindset.suspicions)
                        .sort(([, a], [, b]) => b.confidence - a.confidence)
                        .map(([targetPlayerId, suspicion]) => (
                          <div key={`${observerId}-${targetPlayerId}`} className="suspicion-entry" onClick={() => toggleSuspicionReason(`${observerId}-${targetPlayerId}`)}>
                            <div className="suspicion-summary">
                              <span>{targetPlayerId}</span>
                              <span> - {suspicion.role === 'civilian' ? t('role_civilian') : t('role_spy')}</span>
                              <span> ({ (suspicion.confidence * 100).toFixed(0) }%)</span>
                            </div>
                            {expandedSuspicionReasons[`${observerId}-${targetPlayerId}`] && suspicion.reason && (
                              <div className="suspicion-reason-inline">{t('reason_label')} {suspicion.reason}</div>
                            )}
                          </div>
                        ))}
                    </div>
                  ) : (
                    <p>{t('no_suspicions')}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

      </header>
    </div>
  );
}

export default App;
