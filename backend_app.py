import asyncio, subprocess, uuid, os, re, gc, time
from concurrent.futures import ThreadPoolExecutor

import torch, librosa
from fastapi import FastAPI, UploadFile, Form, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline as hf_pipeline
from llama_cpp import Llama
import llama_cpp

WEBAPP_NCAIR_MODELS = {
    'Hausa':            'NCAIR1/Hausa-ASR',
    'Yoruba':           'NCAIR1/Yoruba-ASR',
    'Igbo':             'NCAIR1/Igbo-ASR',
    'Nigerian English': 'NCAIR1/NigerianAccentedEnglish',
}
WEBAPP_LANG_CODES = {'Hausa': 'ha', 'Yoruba': 'yo', 'Igbo': 'ig', 'Nigerian English': 'en'}
WEBAPP_LANG_NAMES = {'ha': 'Hausa', 'yo': 'Yoruba', 'ig': 'Igbo', 'en': 'Nigerian-accented English'}

CHUNK_SECONDS = 6
EXECUTOR = ThreadPoolExecutor(max_workers=1)  # single GPU: process chunks one at a time

_asr_cache = {}
_llm = None

def get_asr_pipeline(language_name):
    if language_name not in _asr_cache:
        device = 0 if torch.cuda.is_available() else -1
        print(f'[backend] loading ASR model for {language_name} ...')
        _asr_cache[language_name] = hf_pipeline(
            'automatic-speech-recognition',
            model=WEBAPP_NCAIR_MODELS[language_name],
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            generate_kwargs={
                'task': 'transcribe',
                'language': WEBAPP_LANG_CODES[language_name],
                'no_repeat_ngram_size': 3,
                'repetition_penalty': 1.1,
            },
        )
    return _asr_cache[language_name]

def get_llm():
    global _llm
    if _llm is None:
        print('[backend] loading N-ATLaS GGUF (Q4_K_M) ...')
        _llm = Llama.from_pretrained(
            repo_id='tosinamuda/N-ATLaS-GGUF', filename='*Q4_K_M*',
            n_gpu_layers=-1, n_ctx=4096, verbose=False,
        )
        print('  ✅ CUDA offload active.' if llama_cpp.llama_supports_gpu_offload()
              else '  ⚠️  CUDA offload NOT active — this will be slow.')
    return _llm

def build_system_prompt(source_lang_code, do_translate, target_language_name):
    source_name = WEBAPP_LANG_NAMES[source_lang_code]
    fidelity_rule = (
        "CRITICAL: this is a short fragment cut from continuous speech, not a full "
        "sentence. Do NOT invent words, do NOT complete it into a full sentence, and do "
        "NOT add any idea, detail, or fact that is not literally present in the input. "
        "Fix only spelling, obvious mis-heard words, punctuation, and capitalization."
    )
    if not do_translate:
        return (f"You clean up a raw {source_name} speech-to-text fragment. {fidelity_rule} "
                "Never translate. Output ONLY the corrected text, nothing else.")
    return (
        f"You process a raw {source_name} speech-to-text fragment in ONE step:\n"
        f"1) Clean it up. {fidelity_rule}\n"
        f"2) Translate the corrected text into {target_language_name}, staying equally "
        "literal — do not add or complete anything the translation implies but the "
        "original didn't say.\n"
        "Respond in EXACTLY this format and nothing else:\n"
        "CLEANED: <corrected text>\nTRANSLATED: <translation>"
    )

def process_utterance(raw_text, source_lang_code, do_translate, target_language_name, llm):
    if not raw_text.strip():
        return '', ''
    system_prompt = build_system_prompt(source_lang_code, do_translate, target_language_name)
    resp = llm.create_chat_completion(
        messages=[{'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': raw_text}],
        max_tokens=350, repeat_penalty=1.12, temperature=0.1,
    )
    out = resp['choices'][0]['message']['content'].strip()
    if not do_translate:
        return (out if out else raw_text), ''
    m = re.search(r'CLEANED:\s*(.*?)\n\s*TRANSLATED:\s*(.*)', out, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return (out if out else raw_text), ''

app = FastAPI()
SESSIONS = {}  # session_id -> {"captions": [...], "clients": set(), "status": "processing"|"done"}

@app.get('/')
async def root():
    return FileResponse('/content/webapp/static/index.html')

async def broadcast(session_id, message):
    dead = []
    for ws in SESSIONS[session_id]['clients']:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        SESSIONS[session_id]['clients'].discard(ws)

async def process_video(session_id, video_path, source_lang, target_lang, do_translate):
    session = SESSIONS[session_id]
    loop = asyncio.get_event_loop()
    try:
        audio_path = video_path + '_audio.wav'
        await loop.run_in_executor(EXECUTOR, lambda: subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1',
             audio_path, '-loglevel', 'error'], check=True))

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(audio) / sr

        asr = await loop.run_in_executor(EXECUTOR, get_asr_pipeline, source_lang)
        llm = await loop.run_in_executor(EXECUTOR, get_llm)
        source_lang_code = WEBAPP_LANG_CODES[source_lang]

        t = 0.0
        while t < total_duration:
            end = min(t + CHUNK_SECONDS, total_duration)
            chunk = audio[int(t * sr):int(end * sr)]

            result = await loop.run_in_executor(
                EXECUTOR, lambda c=chunk: asr({'raw': c, 'sampling_rate': sr}))
            raw_text = (result.get('text') or '').strip()

            cleaned, translated = await loop.run_in_executor(
                EXECUTOR, process_utterance, raw_text, source_lang_code,
                do_translate, target_lang, llm)

            caption = {'start': round(t, 2), 'end': round(end, 2),
                       'native': cleaned, 'translated': translated}
            session['captions'].append(caption)
            await broadcast(session_id, caption)
            t = end

        session['status'] = 'done'
        await broadcast(session_id, {'done': True})
    except Exception as e:
        print(f'[backend] processing error for {session_id}: {e}')
        await broadcast(session_id, {'done': True, 'error': str(e)})
        session['status'] = 'error'

@app.post('/upload')
async def upload(file: UploadFile = File(...), source_lang: str = Form(...),
                  target_lang: str = Form(...), translate: str = Form(...)):
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {'captions': [], 'clients': set(), 'status': 'processing'}

    os.makedirs('/content/webapp/uploads', exist_ok=True)
    video_path = f'/content/webapp/uploads/{session_id}_{file.filename}'
    with open(video_path, 'wb') as f:
        f.write(await file.read())

    do_translate = str(translate).lower() in ('true', '1', 'yes', 'on')
    asyncio.create_task(process_video(session_id, video_path, source_lang, target_lang, do_translate))

    return {'session_id': session_id}

@app.websocket('/ws/{session_id}')
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in SESSIONS:
        await websocket.close()
        return
    session = SESSIONS[session_id]
    session['clients'].add(websocket)
    for cap in session['captions']:            # replay anything already generated
        await websocket.send_json(cap)
    if session['status'] in ('done', 'error'):
        await websocket.send_json({'done': True})
    try:
        while True:
            await websocket.receive_text()      # just keeps the connection open
    except WebSocketDisconnect:
        session['clients'].discard(websocket)

