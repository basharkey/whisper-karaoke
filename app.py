from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import os
import re
from urllib.parse import unquote
from faster_whisper import WhisperModel

import yt_dlp
from yt_dlp.utils import DownloadError
import ytmusicapi
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel
import shutil
import subprocess


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/tracks/'
app.config['MAX_CONTENT_PATH'] = 1024 * 1024 * 100  # 100 MB limit for file upload

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3'}

def yt_get_song(query: str, songs_dir: Path):
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp.name)

    ydl_opts = {
        'format': 'bestaudio/best',
        # See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': str(tmp_dir.joinpath('%(title)s.%(ext)s'))
    }

    ytm = ytmusicapi.YTMusic()
    for video in ytm.search(query, filter='songs'):
        artist = video['artists'][0]['name']
        album = video['album']['name']
        track = video['title']

        # Add a confirmation in the UI to ask if this is the correct song
        if True:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download(video['videoId'])
                    print(f"https://www.youtube.com/watch?v={video['videoId']}")
                    break
                except DownloadError as e:
                    print(e)

    song_file = list(tmp_dir.glob('*.mp3'))[0]
    demucs_model = 'mdx_extra'
    subprocess.run(['demucs', '-d', 'cuda', '--mp3', '--two-stems', 'vocals', '-o', tmp_dir, '--filename', '{stem}.{ext}', '-n', demucs_model, song_file], check=True)

    lyrics_audio = tmp_dir.joinpath(f'{artist} - {track}.mp3')
    shutil.move(tmp_dir.joinpath(demucs_model, 'vocals.mp3'), lyrics_audio)
    # song_file = song_file.rename(tmp_dir.joinpath(f'{artist} - {track}.mp3'))
    transcribe_file(lyrics_audio)

    shutil.move(tmp_dir.joinpath(demucs_model, 'no_vocals.mp3'), songs_dir.joinpath(f'{artist} - {track}.mp3'))
    for f in tmp_dir.iterdir():
        if f.is_file() and f.suffix == '.txt':
            shutil.move(f, songs_dir)

@app.route('/', methods=["POST"])
def yt_query():
    tracks_dir = Path(app.static_folder, 'tracks')
    query = request.form.get('textbox')

    yt_get_song(query, tracks_dir)
    mp3_files = tracks_dir.rglob('*.mp3')
    return render_template('index.html', mp3_files=mp3_files)

@app.route('/')
def index():
    directory = os.path.join(app.static_folder, 'tracks')
    mp3_files = [f.replace('.mp3', '') for f in os.listdir(directory) if f.endswith('.mp3')]
    return render_template('index.html', mp3_files=mp3_files)

@app.route('/static/tracks/<path:filename>')
def uploaded_file(filename):
    # Decode URL-encoded parts
    decoded_filename = unquote(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], decoded_filename)

    if not os.path.exists(filepath):
        app.logger.error(f"File not found: {filepath}")
        return "File not found", 404

    app.logger.info(f"Serving file: {filepath}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], decoded_filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Use custom sanitization function
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Strip the .mp3 extension from filename before appending text labels
        base_filename = filepath.rsplit('.mp3', 1)[0]
        transcription_path_lines = base_filename + '_lines.txt'
        transcription_path_words = base_filename + '_words.txt'

        if os.path.exists(transcription_path_lines) and os.path.exists(transcription_path_words):
            app.logger.info(f"All transcription files already exist for: {filepath}")
            return jsonify({'message': 'File and transcriptions already exist.', 'filename': filename}), 200

        file.save(filepath)
        app.logger.info(f"File uploaded: {filepath}")
        transcribe_file(filepath)
        return jsonify({'message': 'File uploaded and transcription started!', 'filename': filename})
    else:
        return jsonify({'message': 'No file received or file type not allowed.'}), 400

@app.route('/logs')
def get_logs():
    global log_messages
    messages = log_messages[:]
    log_messages = []  # Clear messages after sending
    return jsonify(messages)

def secure_filename(filename):
    # Remove any path information to avoid directory traversal attacks
    filename = filename.split('/')[-1].split('\\')[-1]
    # Allow only certain characters in filenames to prevent issues on the filesystem - Disabled since we're already taking the name from a file, so it should already be fine.
    # filename = re.sub(r'[^a-zA-Z0-9_.()\[\]-]', '', filename)
    return filename

def transcribe_file(song_file: Path):
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    # Ensure the base filename does not contain '.mp3' for transcription paths
    base_path = song_file.parent
    transcription_path_lines = base_path.joinpath(song_file.stem + '_lines.txt')
    transcription_path_words = base_path.joinpath(song_file.stem + '_words.txt')

    # Check and log file existence
    app.logger.info(f"Checking transcription files: {transcription_path_lines} and {transcription_path_words}")
    if not transcription_path_lines.exists() or not transcription_path_words.exists():
        app.logger.info(f"Transcription started for: {song_file}")
        segments, info = model.transcribe(song_file, beam_size=5, word_timestamps=True)
        with open(transcription_path_lines, 'w', encoding='utf-8') as f_lines, open(transcription_path_words, 'w', encoding='utf-8') as f_words:
            for segment in segments:
                f_lines.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        f_words.write(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}\n")
        app.logger.info(f"Transcription completed for: {song_file}")
    else:
        app.logger.info(f"Transcription files already exist for: {song_file}")

if __name__ == '__main__':
    if shutil.which('ffmpeg'):
        app.run(debug=True, host='0.0.0.0')
    else:
        raise SystemExit("Error: 'ffmpeg' not found in PATH")
