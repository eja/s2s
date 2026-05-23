# S2S

An high-performance, lightweight API server written in Rust that provides local, privacy-conscious Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. By leveraging the `sherpa-onnx` framework, S2S offers state-of-the-art inference with minimal latency, requiring no external cloud dependencies.

The project aims to provide a drop-in local alternative for speech processing, featuring an API structure inspired by industry standards.

## Key Features

- **High Performance:** Built with Rust and ONNX Runtime for efficient CPU/GPU utilization.
- **Privacy-First:** All processing is done locally on your hardware.
- **Automated Model Management:** Built-in bootstrap logic to download and configure necessary models (Kokoro and Parakeet) automatically.
- **Broad STT Language Support:** Supports 25+ languages including English, Spanish, German, French, Russian, and many more.
- **Flexible TTS:** Integration with the **Kokoro** model, supporting over 50 distinct voices across 9 major languages.
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

Alternatively, you can skip the prompts by using the `--auto` flag:

```bash
./s2s --auto
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
| `--auto` | Automatically download missing models | `false` |

---

## API Reference

### 1. Speech-to-Text (STT)
**Endpoint:** `POST /v1/audio/transcriptions`

Transcribe an audio file to text. The endpoint expects a `multipart/form-data` request containing a WAV file. The model automatically detects the language from the supported list.

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

## Language & Voice Support

### Speech-to-Text (STT) Languages
S2S supports transcription for the following languages:

| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| Bulgarian (`bg`) | Croatian (`hr`) | Czech (`cs`) | Danish (`da`) | Dutch (`nl`) |
| English (`en`) | Estonian (`et`) | Finnish (`fi`) | French (`fr`) | German (`de`) |
| Greek (`el`) | Hungarian (`hu`) | Italian (`it`) | Latvian (`lv`) | Lithuanian (`lt`) |
| Maltese (`mt`) | Polish (`pl`) | Portuguese (`pt`) | Romanian (`ro`) | Slovak (`sk`) |
| Slovenian (`sl`) | Spanish (`es`) | Swedish (`sv`) | Russian (`ru`) | Ukrainian (`uk`) |

### Text-to-Speech (TTS) Voices
For TTS, the language is determined automatically based on the prefix of the selected voice.

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
- **Audio Format:** For STT, input must be in **WAV** format (16kHz mono recommended).
- **Disk Space:** Approximately 1.5GB for models and dependencies.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for the underlying inference engine.
- [Kokoro](https://github.com/hexgrad/Kokoro) for the high-quality TTS weights.
- [NVIDIA](https://nvidia.com) for the Parakeet TDT ASR models.
