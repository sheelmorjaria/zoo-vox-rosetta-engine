# Field Deployment - Peer-to-Peer Architecture

This directory contains the deployment configuration for the **Peer-to-Peer** architecture where the Rust Field Engine and Python Cognitive Agent run as independent processes managed by systemd.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Systemd Supervisor                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐     │
│  │  rust-field-engine       │  │  python-cognitive-agent  │     │
│  │  (Technical Architect)   │  │  (Logic Layer)           │     │
│  │                          │  │                          │     │
│  │  - Safety Critical       │  │  - Decision Making       │     │
│  │  - Audio Processing      │◄─┤  - Phrase Selection      │     │
│  │  - Hardware Control      │  │  - Learning              │     │
│  │  - Heartbeat Monitor     │  │  - Intent Generation     │     │
│  │                          │  │                          │     │
│  │  ZeroMQ SUB (Heartbeat)  │◄─┤  ZeroMQ PUB (Heartbeat)  │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

1. **Fail Open to Safety**: If Python crashes, Rust immediately mutes audio and continues in Passthrough Mode
2. **External Supervision**: Systemd handles process lifecycle, not Rust
3. **Peer-to-Peer Communication**: ZeroMQ heartbeats monitor Python availability
4. **State Independence**: Rust starts safely without waiting for Python

## Files

### Systemd Unit Files

- **`rust-field-engine.service`** - Rust execution layer service
- **`python-cognitive-agent.service`** - Python logic layer service (basic)
- **`python-cognitive-agent-with-self-heal.service`** - Python service with self-healing [NEW]

### Python Scripts

- **`python_heartbeat_client.py`** - Example heartbeat client for testing
- **`cognitive_agent_main.py`** - Production cognitive agent with self-healing [NEW]

## Installation

### Option A: Basic Installation (without self-healing)

```bash
# Copy systemd files
sudo cp rust-field-engine.service /etc/systemd/system/
sudo cp python-cognitive-agent.service /etc/systemd/system/

# Adjust paths in service files to match your installation
```

### Option B: Production Installation (with self-healing) [RECOMMENDED]

```bash
# Copy systemd files
sudo cp rust-field-engine.service /etc/systemd/system/
sudo cp python-cognitive-agent-with-self-heal.service /etc/systemd/system/python-cognitive-agent.service

# Copy production cognitive agent
sudo mkdir -p /opt/cognitive
sudo cp cognitive_agent_main.py /opt/cognitive/

# Ensure checkpoint directory exists
sudo mkdir -p /opt/cognitive/state
```

### Adjust Paths

Edit the service files to match your installation:

**`rust-field-engine.service`**:
```ini
ExecStart=/path/to/your/field_engine
```

**`python-cognitive-agent.service`** (with self-healing):
```ini
ExecStart=/usr/bin/python3 /opt/cognitive/cognitive_agent_main.py
WorkingDirectory=/opt/cognitive
Environment="CHECKPOINT_DIR=/opt/cognitive/state"
```

### Reload Systemd

```bash
sudo systemctl daemon-reload
```

### Enable Services

```bash
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service
```

## Usage

### Start Both Services

```bash
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

### Check Status

```bash
# Check Rust Field Engine
sudo systemctl status rust-field-engine.service

# Check Python Cognitive Agent
sudo systemctl status python-cognitive-agent.service

# View logs
sudo journalctl -u rust-field-engine.service -f
sudo journalctl -u python-cognitive-agent.service -f
```

### Stop Services

```bash
sudo systemctl stop python-cognitive-agent.service
sudo systemctl stop rust-field-engine.service
```

## Testing

### Test Heartbeat Client

Run the Python heartbeat client to test connectivity:

```bash
python3 deployment/python_heartbeat_client.py
```

Expected output:
```
============================================================
Python Cognitive Agent - Heartbeat Client
============================================================
Connecting to Rust Field Engine: ipc:///tmp/cognitive_heartbeat.ipc
✓ Connected to Rust Field Engine
  PID: 12345
  Endpoint: ipc:///tmp/cognitive_heartbeat.ipc
Starting heartbeat loop (interval: 20ms)
Press Ctrl+C to stop
Heartbeat sent: sequence=50
Heartbeat sent: sequence=100
...
```

### Simulate Python Crash

```bash
# In one terminal, watch the Rust logs
sudo journalctl -u rust-field-engine.service -f

# In another terminal, kill the Python process
sudo systemctl kill -s SIGKILL python-cognitive-agent.service

# Observe that Rust detects the disconnect and switches to Passthrough Mode
# Systemd will automatically restart Python within 2 seconds
# Rust will detect reconnection and switch back to Interactive Mode
```

## Self-Healing Integration [NEW]

For **long-duration field experiments**, the system includes autonomous crash recovery that preserves state across restarts.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Crash → Systemd Restart → State Recovery → Continue    │
│                                                                  │
│  1. Python crashes (e.g., OOM, exception)                        │
│  2. Rust detects heartbeat loss → Passthrough Mode (safe)        │
│  3. Systemd restarts Python (Restart=always)                    │
│  4. Python loads latest checkpoint → state restored              │
│  5. Python reconnects → Interactive Mode resumed                 │
│  6. Conversation continues with preserved context                │
└─────────────────────────────────────────────────────────────────┘
```

### Installation with Self-Healing

```bash
# Use the self-healing service file instead
sudo cp python-cognitive-agent-with-self-heal.service /etc/systemd/system/python-cognitive-agent.service

# Copy the production cognitive agent
sudo mkdir -p /opt/cognitive
sudo cp cognitive_agent_main.py /opt/cognitive/

# Ensure checkpoint directory exists
sudo mkdir -p /opt/cognitive/state

# Adjust paths in the service file if needed
sudo nano /etc/systemd/system/python-cognitive-agent.service
```

### Service File Configuration

The self-healing service includes additional environment variables:

```ini
[Service]
Environment="RUST_HEARTBEAT_ENDPOINT=ipc:///tmp/cognitive_heartbeat.ipc"
Environment="CHECKPOINT_DIR=/opt/cognitive/state"          # Checkpoint location
Environment="CHECKPOINT_INTERVAL_SEC=60"                    # Save every 60 seconds
```

### Test Self-Healing Workflow

```bash
# In one terminal, monitor Python logs
sudo journalctl -u python-cognitive-agent.service -f

# Observe the startup sequence:
# 1. "Attempting to recover state from checkpoint..."
# 2. "No checkpoint found, starting with fresh state" (first run)
# 3. "✓ Connected to Rust Field Engine"
# 4. "Starting main loop..."

# Kill the Python process to simulate crash
sudo systemctl kill -s SIGKILL python-cognitive-agent.service

# Observe the recovery sequence:
# 1. Systemd restarts Python (within 2 seconds)
# 2. "Attempting to recover state from checkpoint..."
# 3. "✓ Recovered from checkpoint: context=FOOD, history_length=5, turn=3"
# 4. "✓ Connected to Rust Field Engine"
# 5. "Starting main loop..."
```

### Benefits for Field Experiments

**Before Self-Healing:**
- Python crashes → Cold start → Lost context → Confused animals
- Manual intervention required to restore state
- Data loss from interrupted conversations

**After Self-Healing:**
- Python crashes → Warm restart → State restored → Seamless continuation
- No human intervention required
- Full conversation history preserved
- Animals experience minimal disruption
- 16/16 TDD tests ensure reliability

### Checkpoint Contents

Each checkpoint saves:
- **Context**: Current conversation context (e.g., "FOOD", "AGGRESSION")
- **History**: Complete phrase exchange history
- **Dialogue State**: Turn count, initiator, state variables

Example checkpoint (`checkpoint_20250107_120000.json`):
```json
{
  "component": "contextual_agent",
  "context": "FOOD",
  "history": ["PheeA", "PheeB", "ChirpC"],
  "dialogue_state": {
    "turn": 3,
    "initiator": "marmoset"
  },
  "timestamp": "2025-01-07T12:00:00Z"
}
```

## Operation Modes

### Passthrough Mode (Safe Default)

**Conditions**: Python is disconnected or heartbeats have stopped

**Behavior**:
- Audio output is muted
- Raw audio recording continues
- Passive monitoring
- Safe fallback

**Rust Logs**:
```
❌ Cognitive Agent (Python) LOST - Muting Audio
Switching to Passthrough Mode
```

### Interactive Mode (Active)

**Conditions**: Python is connected and sending heartbeats

**Behavior**:
- Processing intents from Python
- Synthesizing responses
- Full cognitive interaction
- Audio output active

**Rust Logs**:
```
⚡ Cognitive Agent (Python) RECONNECTED - PID: 12345
Switching to Interactive Mode
```

## Heartbeat Protocol

### Message Format

```json
{
  "timestamp": 1704067200000,
  "sequence": 123,
  "pid": 12345,
  "state": "active"
}
```

### ZeroMQ Configuration

- **Socket Type**: PUB (Python) → SUB (Rust)
- **Transport**: IPC (Unix Domain Socket)
- **Endpoint**: `ipc:///tmp/cognitive_heartbeat.ipc`
- **Interval**: 20ms (50Hz)
- **Timeout**: 100ms (5 missed heartbeats)

## Troubleshooting

### Python Can't Connect to Rust

1. Check if Rust is running:
   ```bash
   sudo systemctl status rust-field-engine.service
   ```

2. Check if the IPC socket exists:
   ```bash
   ls -la /tmp/cognitive_heartbeat.ipc
   ```

3. Check Rust logs for errors:
   ```bash
   sudo journalctl -u rust-field-engine.service -n 50
   ```

### Python Keeps Crashing

1. Check Python logs for errors:
   ```bash
   sudo journalctl -u python-cognitive-agent.service -n 100
   ```

2. Check for Python syntax/import errors by running manually:
   ```bash
   python3 /opt/cognitive/main.py
   ```

### Rust Not Detecting Python Reconnection

1. Verify Python is sending heartbeats:
   ```bash
   # Monitor with socat (if installed)
   socat - UNIX-CONNECT:/tmp/cognitive_heartbeat.ipc
   ```

2. Check heartbeat interval in Python code (should be < 100ms)

3. Check Rust logs for heartbeat reception:
   ```bash
   sudo journalctl -u rust-field-engine.service | grep -i heartbeat
   ```

## Performance Tuning

### Rust Process Priority

Edit `rust-field-engine.service`:
```ini
# For real-time priority (requires systemd special configuration)
Nice=-10
# Or use CPUSchedulingPolicy=rt for true real-time
```

### Heartbeat Interval

For lower latency, reduce heartbeat interval in `python_heartbeat_client.py`:
```python
HEARTBEAT_INTERVAL_MS = 10  # 10ms = 100Hz
```

Adjust corresponding timeout in Rust `PeerControllerConfig`.

## Security Considerations

1. **IPC Socket Permissions**: The IPC socket is created in `/tmp` which is world-writable. For production, consider:
   - Using a dedicated directory with restricted permissions
   - Using TCP with TLS encryption
   - Using abstract Unix sockets with SO_PASSCRED

2. **Process Isolation**: The systemd unit files use `NoNewPrivileges=true` and `PrivateTmp=true` for security hardening.

3. **Resource Limits**: Consider adding resource limits to prevent runaway processes:
   ```ini
   [Service]
   MemoryMax=1G
   CPUQuota=50%
   ```

## License

CC BY-ND 4.0 International

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)
