// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      realtime.rs
// PATH:      /crates/swiftllm-server/src/api/realtime.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

//! Realtime bidirectional API over WebSocket (`GET /v1/realtime`).
//!
//! Clients open a WebSocket and exchange newline-free JSON events. The protocol
//! is modelled as a pure [`RealtimeSession`] state machine: it ingests client
//! events and returns the server events to emit. The axum handler
//! ([`realtime_ws`]) is a thin transport adapter over that machine, which keeps
//! the protocol fully unit-testable without a live socket.
//!
//! Text streaming is handled end-to-end. Audio frames (binary, or base64 in an
//! `input_audio.append` event) are buffered and their size tracked; speech
//! transcoding is a documented seam — the session reports buffered audio rather
//! than silently pretending to transcribe it.

use crate::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Events sent by the client.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    /// Update session configuration (instructions / modalities).
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// New session settings.
        session: SessionConfig,
    },
    /// Append text to the input buffer.
    #[serde(rename = "input_text.append")]
    InputTextAppend {
        /// Text to append.
        text: String,
    },
    /// Append (base64) audio to the input buffer.
    #[serde(rename = "input_audio.append")]
    InputAudioAppend {
        /// Base64-encoded audio chunk.
        audio: String,
    },
    /// Clear the input buffer.
    #[serde(rename = "input.clear")]
    InputClear,
    /// Ask the server to generate a response from the buffered input.
    #[serde(rename = "response.create")]
    ResponseCreate,
}

/// Mutable session configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionConfig {
    /// System instructions.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Enabled modalities, e.g. `["text"]` or `["text", "audio"]`.
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
}

/// Events emitted by the server.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    /// Sent immediately on connect.
    #[serde(rename = "session.created")]
    SessionCreated {
        /// The session id.
        session_id: String,
    },
    /// Acknowledges a `session.update`.
    #[serde(rename = "session.updated")]
    SessionUpdated {
        /// Active modalities after the update.
        modalities: Vec<String>,
    },
    /// Acknowledges buffered input.
    #[serde(rename = "input.committed")]
    InputCommitted {
        /// Buffered text characters.
        text_chars: usize,
        /// Buffered audio bytes.
        audio_bytes: usize,
    },
    /// A response has started.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// The response id.
        response_id: String,
    },
    /// An incremental text delta.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// The response id.
        response_id: String,
        /// The text fragment.
        delta: String,
    },
    /// The full output text for a response.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        /// The response id.
        response_id: String,
        /// The complete text.
        text: String,
    },
    /// A response has finished.
    #[serde(rename = "response.done")]
    ResponseDone {
        /// The response id.
        response_id: String,
    },
    /// An error occurred.
    #[serde(rename = "error")]
    Error {
        /// Error message.
        message: String,
    },
}

impl ServerEvent {
    /// Serialise the event to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{\"type\":\"error\"}".to_string())
    }
}

/// The realtime protocol state machine.
#[derive(Debug, Clone)]
pub struct RealtimeSession {
    session_id: String,
    instructions: String,
    modalities: Vec<String>,
    input_text: String,
    audio_bytes: usize,
    response_counter: usize,
}

impl RealtimeSession {
    /// Create a new session, returning the session plus the events to emit on
    /// connect (`session.created`).
    pub fn new() -> (Self, Vec<ServerEvent>) {
        let session_id = format!("sess_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);
        let session = Self {
            session_id: session_id.clone(),
            instructions: String::new(),
            modalities: vec!["text".to_string()],
            input_text: String::new(),
            audio_bytes: 0,
            response_counter: 0,
        };
        let events = vec![ServerEvent::SessionCreated { session_id }];
        (session, events)
    }

    /// Append raw audio bytes (used by the binary WebSocket frame path).
    pub fn append_audio_bytes(&mut self, len: usize) -> Vec<ServerEvent> {
        self.audio_bytes += len;
        vec![ServerEvent::InputCommitted {
            text_chars: self.input_text.chars().count(),
            audio_bytes: self.audio_bytes,
        }]
    }

    /// Handle a client event, returning the events to emit in order.
    pub fn handle(&mut self, event: ClientEvent) -> Vec<ServerEvent> {
        match event {
            ClientEvent::SessionUpdate { session } => {
                if let Some(instr) = session.instructions {
                    self.instructions = instr;
                }
                if let Some(modalities) = session.modalities {
                    if modalities.is_empty() {
                        return vec![ServerEvent::Error {
                            message: "modalities must not be empty".to_string(),
                        }];
                    }
                    self.modalities = modalities;
                }
                vec![ServerEvent::SessionUpdated {
                    modalities: self.modalities.clone(),
                }]
            }
            ClientEvent::InputTextAppend { text } => {
                self.input_text.push_str(&text);
                vec![ServerEvent::InputCommitted {
                    text_chars: self.input_text.chars().count(),
                    audio_bytes: self.audio_bytes,
                }]
            }
            ClientEvent::InputAudioAppend { audio } => {
                // The wire form is base64; we track the encoded size rather than
                // decoding (transcoding is a documented seam).
                self.audio_bytes += audio.len();
                vec![ServerEvent::InputCommitted {
                    text_chars: self.input_text.chars().count(),
                    audio_bytes: self.audio_bytes,
                }]
            }
            ClientEvent::InputClear => {
                self.input_text.clear();
                self.audio_bytes = 0;
                vec![ServerEvent::InputCommitted {
                    text_chars: 0,
                    audio_bytes: 0,
                }]
            }
            ClientEvent::ResponseCreate => self.create_response(),
        }
    }

    /// Produce a full response stream for the buffered input.
    fn create_response(&mut self) -> Vec<ServerEvent> {
        self.response_counter += 1;
        let response_id = format!("resp_{}_{}", self.session_id, self.response_counter);
        let mut events = vec![ServerEvent::ResponseCreated {
            response_id: response_id.clone(),
        }];

        let text = self.generate_reply();
        for word in text.split_inclusive(' ') {
            events.push(ServerEvent::OutputTextDelta {
                response_id: response_id.clone(),
                delta: word.to_string(),
            });
        }
        events.push(ServerEvent::OutputTextDone {
            response_id: response_id.clone(),
            text,
        });
        events.push(ServerEvent::ResponseDone { response_id });

        // Consume the input buffer after responding.
        self.input_text.clear();
        self.audio_bytes = 0;
        events
    }

    /// Deterministic stub generation over the buffered input. (The real engine
    /// generation loop is the shared stub boundary across this codebase.)
    fn generate_reply(&self) -> String {
        let chars = self.input_text.chars().count();
        if self.audio_bytes > 0 {
            format!(
                "Received {} text chars and {} audio bytes.",
                chars, self.audio_bytes
            )
        } else if chars == 0 {
            "I have no input buffered yet.".to_string()
        } else {
            format!("You said {} characters; here is my streamed reply.", chars)
        }
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

// ----------------------------------------------------------------------------
// WebSocket transport adapter
// ----------------------------------------------------------------------------

/// `GET /v1/realtime` — upgrade to a realtime WebSocket connection.
pub async fn realtime_ws(ws: WebSocketUpgrade, State(_state): State<AppState>) -> Response {
    ws.on_upgrade(handle_socket)
}

/// Drive a [`RealtimeSession`] over a live socket.
async fn handle_socket(mut socket: WebSocket) {
    let (mut session, initial) = RealtimeSession::new();
    for event in initial {
        if socket.send(Message::Text(event.to_json())).await.is_err() {
            return;
        }
    }

    while let Some(Ok(msg)) = socket.recv().await {
        let events = match msg {
            Message::Text(text) => match serde_json::from_str::<ClientEvent>(&text) {
                Ok(event) => session.handle(event),
                Err(e) => vec![ServerEvent::Error {
                    message: format!("invalid event: {}", e),
                }],
            },
            Message::Binary(bytes) => session.append_audio_bytes(bytes.len()),
            Message::Close(_) => break,
            // Ping/Pong are handled by axum automatically.
            _ => continue,
        };
        for event in events {
            if socket.send(Message::Text(event.to_json())).await.is_err() {
                return;
            }
        }
    }
}

/// Parse a client event from a JSON value (exposed for tooling/tests).
pub fn parse_client_event(value: Value) -> Result<ClientEvent, String> {
    serde_json::from_value(value).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(value: Value) -> ClientEvent {
        parse_client_event(value).unwrap()
    }

    #[test]
    fn new_session_emits_created() {
        let (session, events) = RealtimeSession::new();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ServerEvent::SessionCreated { session_id } => {
                assert_eq!(session_id, session.session_id());
                assert!(session_id.starts_with("sess_"));
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn session_update_changes_modalities() {
        let (mut session, _) = RealtimeSession::new();
        let out = session.handle(ev(json!({
            "type": "session.update",
            "session": {"instructions": "be brief", "modalities": ["text", "audio"]}
        })));
        assert_eq!(
            out,
            vec![ServerEvent::SessionUpdated {
                modalities: vec!["text".into(), "audio".into()]
            }]
        );
        assert_eq!(session.instructions, "be brief");
    }

    #[test]
    fn empty_modalities_is_rejected() {
        let (mut session, _) = RealtimeSession::new();
        let out = session.handle(ev(json!({
            "type": "session.update",
            "session": {"modalities": []}
        })));
        assert!(matches!(out[0], ServerEvent::Error { .. }));
    }

    #[test]
    fn text_append_then_response_streams_deltas() {
        let (mut session, _) = RealtimeSession::new();
        session.handle(ev(json!({"type": "input_text.append", "text": "hello"})));
        let out = session.handle(ev(json!({"type": "response.create"})));

        // First event is response.created, last is response.done.
        assert!(matches!(out.first(), Some(ServerEvent::ResponseCreated { .. })));
        assert!(matches!(out.last(), Some(ServerEvent::ResponseDone { .. })));

        // There is at least one delta and exactly one done-with-text.
        let deltas = out
            .iter()
            .filter(|e| matches!(e, ServerEvent::OutputTextDelta { .. }))
            .count();
        let dones = out
            .iter()
            .filter(|e| matches!(e, ServerEvent::OutputTextDone { .. }))
            .count();
        assert!(deltas >= 1);
        assert_eq!(dones, 1);

        // Reassembling the deltas equals the done text.
        let assembled: String = out
            .iter()
            .filter_map(|e| match e {
                ServerEvent::OutputTextDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        let done_text = out.iter().find_map(|e| match e {
            ServerEvent::OutputTextDone { text, .. } => Some(text.clone()),
            _ => None,
        });
        assert_eq!(Some(assembled), done_text);
    }

    #[test]
    fn input_buffer_is_consumed_after_response() {
        let (mut session, _) = RealtimeSession::new();
        session.handle(ev(json!({"type": "input_text.append", "text": "abc"})));
        session.handle(ev(json!({"type": "response.create"})));
        // Second response now sees an empty buffer.
        let out = session.handle(ev(json!({"type": "response.create"})));
        let done_text = out.iter().find_map(|e| match e {
            ServerEvent::OutputTextDone { text, .. } => Some(text.clone()),
            _ => None,
        });
        assert_eq!(done_text.as_deref(), Some("I have no input buffered yet."));
    }

    #[test]
    fn audio_bytes_tracked_and_reported() {
        let (mut session, _) = RealtimeSession::new();
        let out = session.append_audio_bytes(2048);
        match &out[0] {
            ServerEvent::InputCommitted { audio_bytes, .. } => assert_eq!(*audio_bytes, 2048),
            other => panic!("unexpected: {:?}", other),
        }
        let reply = session.generate_reply();
        assert!(reply.contains("2048 audio bytes"));
    }

    #[test]
    fn input_clear_resets_buffers() {
        let (mut session, _) = RealtimeSession::new();
        session.handle(ev(json!({"type": "input_text.append", "text": "data"})));
        let out = session.handle(ev(json!({"type": "input.clear"})));
        assert_eq!(
            out,
            vec![ServerEvent::InputCommitted { text_chars: 0, audio_bytes: 0 }]
        );
    }

    #[test]
    fn server_event_serializes_with_type_tag() {
        let e = ServerEvent::OutputTextDelta {
            response_id: "resp_1".into(),
            delta: "hi".into(),
        };
        let v: Value = serde_json::from_str(&e.to_json()).unwrap();
        assert_eq!(v["type"], "response.output_text.delta");
        assert_eq!(v["delta"], "hi");
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: realtime.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/src/api/realtime.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
