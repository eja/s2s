// Copyright (C) by Ubaldo Porcheddu <ubaldo@eja.it>

use axum::{
    body::{Body, HttpBody},
    extract::{ConnectInfo, Multipart, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::Deserialize;
use sherpa_onnx::{
    GenerationConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
    OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Cursor};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 35248)]
    port: u16,
    #[arg(long, default_value = "./models")]
    models: PathBuf,
    #[arg(long, default_value = "kokoro-multi-lang-v1_0")]
    kokoro: PathBuf,
    #[arg(long, default_value = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")]
    parakeet: PathBuf,
    #[arg(long, default_value_t = 4)]
    threads: i32,
    #[arg(long)]
    auto: bool,
    #[arg(long)]
    log: Option<PathBuf>,
}

struct AppState {
    recognizer: Option<OfflineRecognizer>,
    tts_engines: RwLock<HashMap<String, OfflineTts>>,
    voice_to_id: HashMap<String, i32>,
    kokoro_path: PathBuf,
    threads: i32,
    has_tts: bool,
}

async fn logger_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let response = next.run(req).await;
    let latency = start.elapsed();
    let status = response.status();
    let size = response.body().size_hint().exact().unwrap_or(0);

    info!(
        "ip:{} method:{} path:{} status:{} size:{} latency:{:?}",
        addr.ip(),
        method,
        path,
        status.as_u16(),
        size,
        latency
    );

    response
}

fn download_and_extract(url: &str, target_dir: &Path, label: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parent = target_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    
    info!("Downloading {} from {}...", label, url);
    let response = reqwest::blocking::get(url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download {}: {}", label, response.status()).into());
    }

    info!("Extracting {}...", label);
    let cursor = Cursor::new(response.bytes()?);
    let bz_decoder = bzip2::read::BzDecoder::new(cursor);
    let mut archive = tar::Archive::new(bz_decoder);
    archive.unpack(parent)?;
    
    info!("Successfully installed {}", label);
    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let stderr_layer = fmt::layer().with_writer(io::stderr);
    
    let file_layer = if let Some(log_path) = &args.log {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .expect("Failed to open log file");
        Some(fmt::layer().with_ansi(false).with_writer(file))
    } else {
        None
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer)
        .with(file_layer)
        .init();

    let kokoro_path = if args.kokoro.is_absolute() {
        args.kokoro.clone()
    } else {
        args.models.join(&args.kokoro)
    };

    let parakeet_path = if args.parakeet.is_absolute() {
        args.parakeet.clone()
    } else {
        args.models.join(&args.parakeet)
    };

    let mut has_tts = kokoro_path.join("model.onnx").exists();
    if !has_tts && args.auto {
        if let Err(e) = download_and_extract(
            "https://github.com/eja/s2s/releases/download/models/kokoro-multi-lang-v1_0.tar.bz2",
            &kokoro_path,
            "Kokoro TTS"
        ) {
            info!("Failed to install Kokoro models: {:?}", e);
        } else {
            has_tts = true;
        }
    }

    let mut has_stt = parakeet_path.join("encoder.int8.onnx").exists();
    if !has_stt && args.auto {
        if let Err(e) = download_and_extract(
            "https://github.com/eja/s2s/releases/download/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
            &parakeet_path,
            "Parakeet STT"
        ) {
            info!("Failed to install Parakeet models: {:?}", e);
        } else {
            has_stt = true;
        }
    }

    if !has_tts && !has_stt {
        eprintln!("Error: Neither Kokoro TTS nor Parakeet STT model is available.");
        eprintln!("Please provide at least one model or use the --auto option to download them.");
        exit(1);
    }

    let recognizer = if has_stt {
        let mut stt_config = OfflineRecognizerConfig::default();
        stt_config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(parakeet_path.join("encoder.int8.onnx").to_string_lossy().into_owned()),
            decoder: Some(parakeet_path.join("decoder.int8.onnx").to_string_lossy().into_owned()),
            joiner: Some(parakeet_path.join("joiner.int8.onnx").to_string_lossy().into_owned()),
        };
        stt_config.model_config.tokens = Some(parakeet_path.join("tokens.txt").to_string_lossy().into_owned());
        stt_config.model_config.num_threads = args.threads;
        OfflineRecognizer::create(&stt_config)
    } else {
        None
    };

    let has_stt = recognizer.is_some();

    if !has_tts && !has_stt {
        eprintln!("Error: Neither Kokoro TTS nor Parakeet STT is available. Exiting.");
        exit(1);
    }

    let voices = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "ff_siwis", "hf_alpha",
        "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi",
        "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao",
        "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
    ];
    let voice_to_id: HashMap<String, i32> = voices.iter().enumerate().map(|(i, &v)| (v.to_string(), i as i32)).collect();

    let state = Arc::new(AppState { 
        recognizer, 
        tts_engines: RwLock::new(HashMap::new()), 
        voice_to_id,
        kokoro_path,
        threads: args.threads,
        has_tts,
    });

    let app = Router::new()
        .route("/", get(landing_page))
        .route("/v1/audio/voices", get(get_voices))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/speech", post(synthesize))
        .layer(middleware::from_fn(logger_middleware))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    info!("Server starting on http://{}", addr);
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await.unwrap();
}

async fn get_voices(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut voices: Vec<String> = state.voice_to_id.keys().cloned().collect();
    voices.sort();
    let voices_json: Vec<serde_json::Value> = voices
        .into_iter()
        .map(|name| serde_json::json!({ "id": name, "name": name }))
        .collect();
    Json(serde_json::json!({ "voices": voices_json })).into_response()
}

async fn transcribe(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> impl IntoResponse {
    let recognizer = match &state.recognizer {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND, "STT service is not available").into_response(),
    };

    let mut audio_bytes = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            if let Ok(bytes) = field.bytes().await { audio_bytes = bytes.to_vec(); }
        }
    }
    if audio_bytes.is_empty() { return (StatusCode::BAD_REQUEST, "No file").into_response(); }

    let cursor = std::io::Cursor::new(audio_bytes);
    let mut reader = match hound::WavReader::new(cursor) {
        Ok(r) => r,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid WAV").into_response(),
    };

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
        },
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    };

    let samples = if channels > 1 {
        raw_samples.chunks_exact(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    let stream = recognizer.create_stream();
    stream.accept_waveform(spec.sample_rate as i32, &samples);
    recognizer.decode(&stream);
    Json(serde_json::json!({ "text": stream.get_result().map(|r| r.text).unwrap_or_default() })).into_response()
}

#[derive(Deserialize)]
struct SpeechRequest {
    input: String,
    voice: Option<String>,
}

async fn synthesize(State(state): State<Arc<AppState>>, Json(payload): Json<SpeechRequest>) -> impl IntoResponse {
    if !state.has_tts {
        return (StatusCode::NOT_FOUND, "TTS service is not available").into_response();
    }

    let voice_name = payload.voice.unwrap_or_else(|| "af_alloy".to_string());
    let sid = state.voice_to_id.get(&voice_name).cloned().unwrap_or(0);
    
    let lang_code = match voice_name.chars().next() {
        Some('a') => "en-us",
        Some('b') => "en-gb",
        Some('e') => "es",
        Some('f') => "fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('j') => "ja",
        Some('p') => "pt-br",
        Some('z') => "zh",
        _ => "en-us",
    };

    {
        let engines = state.tts_engines.read().await;
        if !engines.contains_key(lang_code) {
            drop(engines);
            let mut engines_mut = state.tts_engines.write().await;
            if !engines_mut.contains_key(lang_code) {
                let lexicon_file = match lang_code {
                    "en-us" => "lexicon-us-en.txt",
                    "en-gb" => "lexicon-gb-en.txt",
                    "zh" => "lexicon-zh.txt",
                    _ => "",
                };

                let mut tts_config = OfflineTtsConfig::default();
                tts_config.model.num_threads = state.threads;
                tts_config.model.kokoro = OfflineTtsKokoroModelConfig {
                    model: Some(state.kokoro_path.join("model.onnx").to_string_lossy().into_owned()),
                    voices: Some(state.kokoro_path.join("voices.bin").to_string_lossy().into_owned()),
                    tokens: Some(state.kokoro_path.join("tokens.txt").to_string_lossy().into_owned()),
                    data_dir: Some(state.kokoro_path.join("espeak-ng-data").to_string_lossy().into_owned()),
                    dict_dir: Some(state.kokoro_path.join("dict").to_string_lossy().into_owned()),
                    lexicon: Some(if lexicon_file.is_empty() { String::new() } else { state.kokoro_path.join(lexicon_file).to_string_lossy().into_owned() }),
                    lang: Some(lang_code.to_string()),
                    length_scale: 1.0,
                };
                if let Some(tts) = OfflineTts::create(&tts_config) {
                    engines_mut.insert(lang_code.to_string(), tts);
                }
            }
        }
    }

    let engines = state.tts_engines.read().await;
    let tts = match engines.get(lang_code) {
        Some(e) => e,
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "Engine failed to load").into_response(),
    };

    let mut gen_config = GenerationConfig::default();
    gen_config.sid = sid;
    gen_config.speed = 1.0;

    let audio = match tts.generate_with_config::<fn(&[f32], f32) -> bool>(&payload.input, &gen_config, None) {
        Some(a) => a,
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "TTS Generation failed").into_response(),
    };

    let mut buf = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate() as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
    for &s in audio.samples() { writer.write_sample(s).unwrap(); }
    writer.finalize().unwrap();

    axum::response::Response::builder()
        .header("Content-Type", "audio/wav")
        .body(axum::body::Body::from(buf.into_inner()))
        .unwrap()
}

async fn landing_page() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S2S</title>
    <style>
        body { max-width: 600px; margin: 40px auto; padding: 20px; color: #333; }
        section { background: #f7f7f7; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #ddd; }
        form { display: grid; gap: 10px; }
        input, select, button { padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }
        button { background: #0070f3; color: white; border: none; cursor: pointer; }
        audio { width: 100%; margin-top: 10px; }
    </style>
</head>
<body>
    <section>
        <h2>Speech to Text</h2>
        <form id="asr-form">
            <input type="file" id="audio-file" accept="audio/wav" required>
            <button type="submit">Transcribe</button>
        </form>
        <div id="asr-result" style="margin-top:10px; word-break:break-all;"></div>
    </section>
    <section>
        <h2>Text to Speech</h2>
        <form id="tts-form">
            <input type="text" id="tts-text" value="Hello, welcome to speech services." required>
            <select id="tts-voice"></select>
            <button type="submit">Synthesize</button>
        </form>
        <audio id="tts-audio" controls style="display:none;"></audio>
    </section>
    <script>
        function getRelativeUrl(path) {
            const base = window.location.pathname.endsWith('/') 
                ? window.location.pathname 
                : window.location.pathname + '/';
            return base + path;
        }

        async function loadVoices() {
            try {
                const res = await fetch(getRelativeUrl('v1/audio/voices'));
                if (!res.ok) throw new Error('Failed to fetch voices');
                const data = await res.json();
                const select = document.getElementById('tts-voice');
                select.innerHTML = '';
                data.voices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.id;
                    opt.textContent = v.name;
                    select.appendChild(opt);
                });
            } catch (err) {
                console.error('Error loading voices:', err);
            }
        }
        loadVoices();

        document.getElementById('asr-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('audio-file');
            if (!fileInput.files.length) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const resultDiv = document.getElementById('asr-result');
            resultDiv.textContent = 'Transcribing...';
            try {
                const res = await fetch(getRelativeUrl('v1/audio/transcriptions'), { method: 'POST', body: formData });
                if (!res.ok) {
                    throw new Error(res.status === 404 ? 'STT service is not available' : 'Transcription failed');
                }
                const json = await res.json();
                resultDiv.textContent = json.text || 'No transcription result.';
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err.message;
            }
        });
        document.getElementById('tts-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('tts-text').value;
            const voice = document.getElementById('tts-voice').value;
            const audio = document.getElementById('tts-audio');
            audio.style.display = 'none';
            try {
                const res = await fetch(getRelativeUrl('v1/audio/speech'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input: text, voice })
                });
                if (!res.ok) {
                    throw new Error(res.status === 404 ? 'TTS service is not available' : 'Generation failed');
                }
                const blob = await res.blob();
                audio.src = URL.createObjectURL(blob);
                audio.style.display = 'block';
                audio.play();
            } catch (err) {
                alert('Error: ' + err.message);
            }
        });
    </script>
</body>
</html>"#)
}
