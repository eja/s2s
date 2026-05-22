# S2S

An high-performance, lightweight API server written in Rust that provides local, privacy-conscious Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. By leveraging the `sherpa-onnx` framework, S2S offers state-of-the-art inference with minimal latency, requiring no external cloud dependencies.

The project aims to provide a drop-in local alternative for speech processing, featuring an API structure inspired by industry standards.

## Key Features

- **High Performance:** Built with Rust and ONNX Runtime for efficient CPU/GPU utilization.
- **Privacy-First:** All processing is done locally on your hardware.
- **Automated Model Management:** Built-in bootstrap logic to download and configure necessary models (Kokoro and Parakeet) automatically.
- **Multi-Language Support:** Comprehensive support for English (US/UK), Chinese, Spanish, French, Hindi, Italian, Japanese, and Portuguese.
- **Flexible TTS:** Integration with the **Kokoro** model, supporting over 50 distinct voices.
- **Robust STT:** Powered by the **NVIDIA Parakeet TDT** model for highly accurate transcriptions.

---

## Getting Started

### Installation
Download the latest executable for your platform from the [Releases](https://github.com/eja/s2s/releases) page.

### Running the Server
Simply run the executable to start the server. On the first run, the application will ask for permission to download the required ONNX models (~1GB total).

```bash
./s2s
```

Alternatively, you can skip the prompts by using the `--download` flag:

```bash
./s2s --download
```

### Configuration Options
The server can be customized via command-line arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--host` | The IP address to bind the server to | `127.0.0.1` |
| `--port` | The port to listen on | `35248` |
| `--kokoro` | Path to the Kokoro TTS model directory | `./models/kokoro...` |
| `--parakeet` | Path to the Parakeet STT model directory | `./models/sherpa...` |
| `--threads` | Number of threads for inference | `4` |
| `--download` | Automatically download missing models | `false` |

---

## API Reference

### 1. Speech-to-Text (STT)
**Endpoint:** `POST /v1/audio/transcriptions`

Transcribe an audio file to text. The endpoint expects a `multipart/form-data` request containing a WAV file.

**Request:**
```bash
curl http://127.0.0.1:35248/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "text": "Hello world, this is a local transcription."
}
```

### 2. Text-to-Speech (TTS)
**Endpoint:** `POST /v1/audio/speech`

Synthesize text into high-quality audio.

**Request Body:**
| Field | Type | Description |
| :--- | :--- | :--- |
| `input` | String | The text to be synthesized |
| `voice` | String | (Optional) The voice ID (Default: `af_alloy`) |

**Example:**
```bash
curl http://127.0.0.1:35248/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, I am a locally hosted voice.",
    "voice": "af_bella"
  }' --output output.wav
```

---

## Supported Voices and Languages

S2S determines the language automatically based on the prefix of the selected voice.

| Language | Voice Prefix | Examples |
| :--- | :--- | :--- |
| **English (US)** | `af_`, `am_` | `af_alloy`, `af_sky`, `am_adam`, `am_echo` |
| **English (UK)** | `bf_`, `bm_` | `bf_alice`, `bm_daniel` |
| **Spanish** | `ef_`, `em_` | `ef_dora`, `em_alex` |
| **French** | `ff_` | `ff_siwis` |
| **Hindi** | `hf_`, `hm_` | `hf_alpha`, `hm_psi` |
| **Italian** | `if_`, `im_` | `if_sara`, `im_nicola` |
| **Japanese** | `jf_`, `jm_` | `jf_alpha`, `jm_kumo` |
| **Portuguese** | `pf_`, `pm_` | `pf_dora`, `pm_santa` |
| **Chinese** | `zf_`, `zm_` | `zf_xiaobei`, `zm_yunxi` |

---

## Requirements

- **Operating System:** Linux, macOS, or Windows.
- **Audio Format:** For STT, input must be in **WAV** format.
- **Disk Space:** Approximately 1.5GB for models and dependencies.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for the underlying inference engine.
- [Kokoro](https://github.com/hexgrad/Kokoro) for the high-quality TTS weights.
- [NVIDIA](https://nvidia.com) for the Parakeet TDT ASR models.
