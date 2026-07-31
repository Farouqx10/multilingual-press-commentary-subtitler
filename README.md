# Group 2: Multi-Lingual Press Commentary / Movie Subtitler

## Project Overview

The Multi-Lingual Press Commentary / Movie Subtitler is an AI-powered application designed to automatically generate subtitles for Nigerian press news videos and indigenous movies. The system leverages Automatic Speech Recognition (ASR), Large Language Models (LLMs), and Machine Translation to produce accurate, readable, and multilingual subtitles.

This project was developed as part of the NCAIR AI Training Programme.

---

## Objectives

- Automatically convert speech in videos into text.
- Generate synchronized subtitles.
- Improve transcript quality using a Large Language Model.
- Translate subtitles into multiple Nigerian languages.
- Provide a simple interface for processing videos.

---

## Features

- Automatic Speech Recognition (ASR)
- Subtitle generation
- Subtitle refinement using N-ATLAS
- Translation support
- Video subtitle preview
- Offline processing capability
- Easy-to-use graphical interface

---

## Technologies Used

- Python
- Google Colab
- Hugging Face Transformers
- NCAIR ASR Models
- N-ATLAS LLM
- FFmpeg
- Librosa
- PySRT
- SoundFile
- Accelerate

---

## Project Workflow

1. Upload a video.
2. Extract the audio from the video.
3. Convert speech to text using the NCAIR ASR model.
4. Refine and correct the transcript using N-ATLAS.
5. Translate the transcript (optional).
6. Generate subtitle (.srt) files.
7. Merge subtitles with the original video.
8. Export the final subtitled video.

---

Video Input
    │
    ▼
Language Identification
    │
    ▼
User confirms source language + picks target language
    │
    ▼
Stage 1 — Transcription
    NCAIR ASR model (Hausa / Yoruba / Igbo / Nigerian English)
    → chunked inference (30s windows, 2s overlap)
    → word-level merge & de-duplication
    → subtitle block segmentation (duration/pause/word-count/punctuation)
    → raw .srt
    │
    ▼
Stage 2 — Cleanup (N-ATLaS GGUF, llama.cpp, batched w/ retry+validation)
    Spelling/punctuation/ASR-error correction only, no rewriting
    → cleaned .srt
    │
    ▼
Stage 3 — Translation (N-ATLaS GGUF, llama.cpp, batched w/ retry+validation)
    Dynamic source→target prompt (any of the 4 langs + English)
    → bilingual .srt (original line + translated line)
    │
    ▼
Burn-in (ffmpeg subtitles filter, resolution-scaled font)
    → final captioned .mp4
    │
    ▼
(Optional) Real-time Web App — FastAPI backend + vanilla HTML/JS frontend,
tunneled via ngrok for a public Colab-hosted URL

---

## Repository Structure

Multi-Lingual-Press-Commentary-Subtitler/
│
├── README.md
├── LICENSE
├── .gitignore
└── Multi_Lingual_Subtitler.ipynb

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install the required packages manually:

```bash
pip install transformers accelerate librosa soundfile pysrt ipywidgets hf_transfer llama-cpp-python
```

---

## Usage

1. Open the notebook in Google Colab.
2. Install the required dependencies.
3. Upload the input video.
4. Run each notebook cell sequentially.
5. Download the generated subtitle file and subtitled video.

---

## Sample Output

The application produces:

- Accurate transcript
- Subtitle (.srt) file
- Final video with embedded subtitles

---

## Future Improvements

- Support additional Nigerian languages.
- Real-time subtitle generation.
- Speaker identification.
- Improved subtitle synchronization.
- Desktop application packaging.
- GPU optimization for faster inference.

---

## Acknowledgements

We sincerely acknowledge:

- **National Centre for Artificial Intelligence and Robotics (NCAIR)** for providing the training programme, learning resources, and project guidance.
- **Our Facilitators** for their mentorship, technical support, and continuous guidance throughout the project.
- **All members of our project group** for their collaboration, dedication, and contributions toward the successful completion of this project.

---

## Group Members
- Daniel Ottah
- Titlayoomi Kehinde
- Nzubechukwu Illo
- Umar Ibrahim 
- Isaac Famiyesin
- Imran Ibrahim
- Chiedozie Chimah
- David Olaniyi
  
---

## Facilitators
- Victor Rizama
- Stephen Ayuba

---

## License
MIT

This project was developed for educational purposes as part of the NCAIR AI Training Programme.
