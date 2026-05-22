// Copyright (C) by Ubaldo Porcheddu <ubaldo@eja.it>

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
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
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 35248)]
    port: u16,
    #[arg(long, default_value = "./models/kokoro-multi-lang-v1_0")]
    kokoro: PathBuf,
    #[arg(long, default_value = "./models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")]
    parakeet: PathBuf,
    #[arg(long, default_value_t = 4)]
    threads: i32,
    #[arg(long)]
    download: bool,
}

struct AppState {
    recognizer: OfflineRecognizer,
    tts_engines: RwLock<HashMap<String, OfflineTts>>,
    voice_to_id: HashMap<String, i32>,
    kokoro_path: PathBuf,
    threads: i32,
}

fn ask_user(model_name: &str) -> bool {
    print!("Model '{}' not found. Would you like to download it? (y/N): ", model_name);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_lowercase() == "y"
}

fn download_and_extract(url: &str, target_dir: &Path, label: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parent = target_dir.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    
    println!("Downloading {}...", label);
    let response = reqwest::blocking::get(url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download {}: {}", label, response.status()).into());
    }

    println!("Extracting {}...", label);
    let cursor = Cursor::new(response.bytes()?);
    let bz_decoder = bzip2::read::BzDecoder::new(cursor);
    let mut archive = tar::Archive::new(bz_decoder);
    archive.unpack(parent)?;
    
    println!("Successfully installed {}", label);
    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if !args.kokoro.join("model.onnx").exists() {
        if args.download || ask_user("Kokoro TTS") {
            download_and_extract(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2",
                &args.kokoro,
                "Kokoro TTS"
            ).expect("Failed to install Kokoro models");
        } else {
            exit(1);
        }
    }

    if !args.parakeet.join("encoder.int8.onnx").exists() {
        if args.download || ask_user("Parakeet STT") {
            download_and_extract(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
                &args.parakeet,
                "Parakeet STT"
            ).expect("Failed to install Parakeet models");
        } else {
            exit(1);
        }
    }

    let mut stt_config = OfflineRecognizerConfig::default();
    stt_config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(args.parakeet.join("encoder.int8.onnx").to_string_lossy().into_owned()),
        decoder: Some(args.parakeet.join("decoder.int8.onnx").to_string_lossy().into_owned()),
        joiner: Some(args.parakeet.join("joiner.int8.onnx").to_string_lossy().into_owned()),
    };
    stt_config.model_config.tokens = Some(args.parakeet.join("tokens.txt").to_string_lossy().into_owned());
    stt_config.model_config.num_threads = args.threads;
    let recognizer = OfflineRecognizer::create(&stt_config).expect("STT init failed");

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
        kokoro_path: args.kokoro,
        threads: args.threads,
    });

    let app = Router::new()
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/speech", post(synthesize))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("Server running on http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn transcribe(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> impl IntoResponse {
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

    let stream = state.recognizer.create_stream();
    stream.accept_waveform(spec.sample_rate as i32, &samples);
    state.recognizer.decode(&stream);
    Json(serde_json::json!({ "text": stream.get_result().map(|r| r.text).unwrap_or_default() })).into_response()
}

#[derive(Deserialize)]
struct SpeechRequest {
    input: String,
    voice: Option<String>,
}

async fn synthesize(State(state): State<Arc<AppState>>, Json(payload): Json<SpeechRequest>) -> impl IntoResponse {
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
