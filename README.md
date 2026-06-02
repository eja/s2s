# S2S

A high-performance, lightweight API server written in Rust that provides local, privacy-conscious Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities. By leveraging the `sherpa-onnx` framework, S2S offers efficient local inference with minimal latency, requiring no external cloud dependencies.

The project aims to provide a drop-in local alternative for speech processing, featuring an API structure inspired by industry standards.

## Key Features

- **Local Inference:** All processing is done locally on your hardware.
- **Request Tracing:** Integrated logging providing real-time insights into IP addresses, status codes, and request latency.
- **Automated Model Management:** Built-in bootstrap logic to download necessary models automatically when using the `--auto` flag.
- **Flexible Service Fallbacks:** The server starts as long as at least one model is present. If only one model is loaded, requests to the missing service will return `404 Not Found`.
- **OpenAI-Compatible Voice Directory:** Exposes a standard `/v1/audio/voices` list, allowing client integrations to discover voices dynamically.
- **Broad STT Language Support:** Supports 25+ languages including English, Spanish, German, French, Russian, and many more.
- **Flexible TTS:** Integration with the **Kokoro** model, supporting over 50 distinct voices across 9 major languages.
- **Robust STT:** Powered by the **NVIDIA Parakeet TDT** model for accurate transcriptions.

---

## Getting Started

### Installation
Download the latest executable for your platform from the [Releases](https://github.com/eja/s2s/releases) page.

### Running the Server
The application requires at least one of the two models to be present locally in order to run. Execute the binary to start the server:

```bash
./s2s
```

If neither model is found on your system, the server will inform you and exit. You can instruct the server to automatically download and configure the required ONNX models (~1GB total) by specifying the `--auto` flag:

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
| `--log` | Path to a file for persistent logging | `stderr` |

---

## API Reference

> **Note:** If the TTS or STT model is missing at startup, the server still launches, but any requests to the missing endpoints will return `404 Not Found`.

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

Synthesize text into audio.

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

### 3. Voice Discovery
**Endpoint:** `GET /v1/audio/voices`

Retrieve the list of available TTS voices sorted alphabetically.

**Request:**
```bash
curl http://127.0.0.1:35248/v1/audio/voices
```

**Response:**
```json
{
  "voices": [
    { "id": "af_alloy", "name": "af_alloy" },
    { "id": "af_aoede", "name": "af_aoede" }
  ]
}
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
- [Kokoro](https://github.com/hexgrad/Kokoro) for the TTS weights.
- [NVIDIA](https://nvidia.com) for the Parakeet TDT ASR models.
