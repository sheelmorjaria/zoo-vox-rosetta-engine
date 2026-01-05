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
- **`python-cognitive-agent.service`** - Python logic layer service

### Python Scripts

- **`python_heartbeat_client.py`** - Example heartbeat client for testing

## Installation

### 1. Copy Systemd Files

```bash
sudo cp rust-field-engine.service /etc/systemd/system/
sudo cp python-cognitive-agent.service /etc/systemd/system/
```

### 2. Adjust Paths

Edit the service files to match your installation:

**`rust-field-engine.service`**:
```ini
ExecStart=/path/to/your/field_engine
```

**`python-cognitive-agent.service`**:
```ini
ExecStart=/usr/bin/python3 /path/to/your/cognitive_agent.py
WorkingDirectory=/path/to/your/cognitive/directory
```

### 3. Reload Systemd

```bash
sudo systemctl daemon-reload
```

### 4. Enable Services

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
