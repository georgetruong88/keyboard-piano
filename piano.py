#!/usr/bin/env python3
"""
Keyboard Piano - play musical notes with your laptop keyboard.

Controls:
  A S D F G H J K   -> C D E F G A B C (white keys, one octave)
  W E   T Y U       -> C# D#   F# G# A# (black keys)
  Z / X             -> shift octave down / up
  Space (hold)      -> sustain (longer note decay)
  1-9, 0            -> instrument: sine / square / sawtooth / triangle / guitar /
                       drum synth / pipa / guzheng / harmonica /
                       Chinese bamboo flute (dizi)
  ,                 -> instrument: electric fire guitar (overdriven/distorted)
  .                 -> instrument: DJ turntable scratch
  /                 -> scratch stab: one-shot DJ scratch hit that layers on top
                       of whatever instrument is selected and the backing beat,
                       without switching your current instrument
  - / =             -> volume down / up
  M                 -> mute / unmute
  R                 -> start/stop recording
  P                 -> play back last recording
  SAVE / LOAD       -> click the on-screen buttons to persist a recording to disk
                       and reload it in a later session
  B                 -> start/stop backing beat
  N                 -> cycle beat pattern (Rock, Four on the Floor, Hip-Hop,
                       Funk, Reggae, Trap, plus epic/historical war-drum
                       patterns: Taiko, War March, Mongol Gallop,
                       Viking War Drum, Ottoman Mehter)
  [ / ]             -> tempo down / up
  Ctrl + F1..F5     -> save current instrument + beat pattern + tempo as a preset
  F1..F5            -> load that preset (also persisted to presets.json on disk)
  Esc               -> quit
"""

import json
import math
import os
import time
from collections import deque

# Disable IME hooking (e.g. IBus/Unikey) for this process only, before SDL/X11
# init. Without this, input methods that treat plain letters as compose keys
# (Vietnamese Telex uses a/s/d/f/g/h/... exactly like our note keys) can
# silently swallow every KEYDOWN before pygame ever sees it.
os.environ["XMODIFIERS"] = "@im=none"
os.environ["SDL_IME_SHOW_UI"] = "0"

import numpy as np
import pygame

# ---------------------------------------------------------------- audio ----

SAMPLE_RATE = 44100
pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=256)
pygame.init()
pygame.mixer.set_num_channels(32)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# key -> semitone offset from C in the current octave (C..C of next octave)
KEY_OFFSETS = {
    pygame.K_a: 0,
    pygame.K_w: 1,
    pygame.K_s: 2,
    pygame.K_e: 3,
    pygame.K_d: 4,
    pygame.K_f: 5,
    pygame.K_t: 6,
    pygame.K_g: 7,
    pygame.K_y: 8,
    pygame.K_h: 9,
    pygame.K_u: 10,
    pygame.K_j: 11,
    pygame.K_k: 12,
}

WAVEFORMS = {
    pygame.K_1: "sine",
    pygame.K_2: "square",
    pygame.K_3: "sawtooth",
    pygame.K_4: "triangle",
    pygame.K_5: "guitar",
    pygame.K_6: "drums",
    pygame.K_7: "pipa",
    pygame.K_8: "guzheng",
    pygame.K_9: "harmonica",
    pygame.K_0: "dizi",
    pygame.K_COMMA: "fire_guitar",
    pygame.K_PERIOD: "dj_scratch",
}

INSTRUMENT_LABELS = {
    "sine": "sine",
    "square": "square",
    "sawtooth": "sawtooth",
    "triangle": "triangle",
    "guitar": "guitar",
    "drums": "drum synth",
    "harmonica": "harmonica",
    "dizi": "Chinese bamboo flute (dizi)",
    "fire_guitar": "electric fire guitar",
    "pipa": "pipa",
    "guzheng": "guzheng",
    "dj_scratch": "DJ turntable scratch",
}

REVERSE_KEY_OFFSETS = {offset: key for key, offset in KEY_OFFSETS.items()}


def note_name(octave, offset):
    idx = offset % 12
    oct_up = octave + offset // 12
    return f"{NOTE_NAMES[idx]}{oct_up}"


def midi_number(octave, offset):
    # MIDI: C4 = 60
    return 12 * (octave + 1) + offset


def freq_for(octave, offset):
    n = midi_number(octave, offset)
    return 440.0 * (2 ** ((n - 69) / 12))


def make_basic_wave(freq, duration, shape):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    if shape == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif shape == "sawtooth":
        wave = 2 * (t * freq - np.floor(0.5 + t * freq))
    elif shape == "triangle":
        wave = 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
    else:
        wave = np.sin(2 * np.pi * freq * t)

    # simple percussive envelope: fast attack, exponential decay
    attack = int(0.01 * SAMPLE_RATE)
    envelope = np.ones_like(wave)
    envelope[:attack] = np.linspace(0, 1, attack)
    decay_curve = np.exp(-3.0 * t / duration)
    envelope *= decay_curve

    wave = wave * envelope * 0.3  # headroom so chords don't clip
    audio = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def karplus_strong_core(freq, duration, decay=0.995, damping=0.5):
    """Plucked-string model: a decaying noise loop through a lowpass filter.
    Lower damping keeps more high end (brighter/twangier); decay closer to 1
    rings out longer (more resonant string)."""
    n_samples = int(SAMPLE_RATE * duration)
    period = max(2, int(round(SAMPLE_RATE / freq)))
    buf = deque(np.random.uniform(-1, 1, period))
    out = np.empty(n_samples)
    for i in range(n_samples):
        out[i] = buf[0]
        avg = decay * (damping * buf[0] + (1 - damping) * buf[1])
        buf.append(avg)
        buf.popleft()
    return out


def make_guitar_wave(freq, duration):
    out = karplus_strong_core(freq, duration, decay=0.995, damping=0.5)
    n_samples = len(out)
    attack = int(0.002 * SAMPLE_RATE)
    envelope = np.ones(n_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    wave = out * envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_fire_guitar_wave(freq, duration):
    """Electric guitar, overdriven: a Karplus-Strong string pushed through
    tanh waveshaping for distortion grit, plus a thin amp-fizz noise layer,
    with longer sustain than the clean guitar (distortion compresses/sustains)."""
    out = karplus_strong_core(freq, duration, decay=0.9975, damping=0.42)
    n_samples = len(out)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    driven = np.tanh(out * 6.0)
    fizz = np.random.uniform(-1, 1, n_samples) * 0.03 * np.exp(-2.0 * t / duration)
    wave = driven * 0.8 + fizz

    attack = int(0.001 * SAMPLE_RATE)
    envelope = np.ones(n_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    wave *= envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.55 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_dj_scratch_wave(freq, duration):
    """DJ turntable scratch: a triangle-LFO sweeps playback speed back and
    forth like a hand working a record, driving a harsh square-ish tone;
    amplitude gates hardest at the direction reversals (the classic scratch
    'chirp'), with vinyl surface noise mixed underneath."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    scratch_rate = 8.0  # back-and-forth strokes per second
    lfo = 2 * np.abs(2 * (t * scratch_rate - np.floor(0.5 + t * scratch_rate))) - 1  # triangle, -1..1
    pitch_mult = np.clip(1 + 0.9 * lfo, 0.1, None)
    inst_freq = freq * pitch_mult
    phase = 2 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE
    tone = np.sign(np.sin(phase))
    noise = np.random.uniform(-1, 1, n) * 0.25
    gate = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * scratch_rate * t))
    wave = (tone * 0.7 + noise) * gate
    wave *= np.exp(-1.0 * t / duration)
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_pipa_wave(freq, duration):
    """Pipa: bright, percussive plectrum pluck with a fast-decaying string."""
    out = karplus_strong_core(freq, duration, decay=0.988, damping=0.32)
    n_samples = len(out)
    click_len = min(int(0.006 * SAMPLE_RATE), n_samples)
    click = np.zeros(n_samples)
    click[:click_len] = np.random.uniform(-1, 1, click_len) * np.exp(
        -40.0 * np.linspace(0, 1, click_len)
    )
    attack = int(0.001 * SAMPLE_RATE)
    envelope = np.ones(n_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    wave = out * envelope + click * 0.5
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_guzheng_wave(freq, duration):
    """Guzheng: long resonant zither string, doubled and slightly detuned
    for the shimmering chorus-like ring characteristic of its metal strings."""
    out1 = karplus_strong_core(freq, duration, decay=0.9985, damping=0.5)
    out2 = karplus_strong_core(freq * 1.003, duration, decay=0.9985, damping=0.5)
    n_samples = len(out1)
    attack = int(0.003 * SAMPLE_RATE)
    envelope = np.ones(n_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    wave = (out1 * 0.65 + out2 * 0.45) * envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_drum_wave(freq, duration):
    """Tuned percussion (steel-drum-like): fundamental plus fast-decaying overtones."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-6.0 * t / duration)
        + 0.3 * np.sin(2 * np.pi * freq * 3.4 * t) * np.exp(-10.0 * t / duration)
    )
    attack = int(0.005 * SAMPLE_RATE)
    envelope = np.ones_like(wave)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope *= np.exp(-5.0 * t / duration)
    wave *= envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_dizi_wave(freq, duration):
    """Chinese bamboo flute (dizi): bright overtones plus the characteristic
    dimo membrane buzz - a fast, shallow AM rasp not present in a plain flute."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    vibrato = 1 + 0.015 * np.sin(2 * np.pi * 7 * t)
    tone = (
        np.sin(2 * np.pi * freq * vibrato * t)
        + 0.35 * np.sin(2 * np.pi * freq * 2 * vibrato * t)
        + 0.22 * np.sin(2 * np.pi * freq * 3 * vibrato * t)
        + 0.10 * np.sin(2 * np.pi * freq * 4 * vibrato * t)
    )
    membrane_buzz = 1 + 0.18 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t)
    breath = np.random.uniform(-1, 1, len(t)) * 0.03
    wave = tone * membrane_buzz * 0.8 + breath

    attack = min(int(0.03 * SAMPLE_RATE), len(wave))
    envelope = np.ones_like(wave)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope *= np.exp(-1.1 * t / duration)
    wave *= envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.4 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_harmonica_wave(freq, duration):
    """Reedy harmonica: buzzy odd-harmonic stack with tremolo."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tremolo = 1 + 0.15 * np.sin(2 * np.pi * 7 * t)
    wave = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * freq * 3 * t)
        + 0.3 * np.sin(2 * np.pi * freq * 5 * t)
        + 0.2 * np.sin(2 * np.pi * freq * 7 * t)
    )
    wave *= tremolo

    attack = int(0.02 * SAMPLE_RATE)
    envelope = np.ones_like(wave)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope *= np.exp(-1.5 * t / duration)
    wave *= envelope
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.35 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_wave(freq, duration, shape):
    if shape == "guitar":
        return make_guitar_wave(freq, duration)
    if shape == "fire_guitar":
        return make_fire_guitar_wave(freq, duration)
    if shape == "drums":
        return make_drum_wave(freq, duration)
    if shape == "harmonica":
        return make_harmonica_wave(freq, duration)
    if shape == "dizi":
        return make_dizi_wave(freq, duration)
    if shape == "pipa":
        return make_pipa_wave(freq, duration)
    if shape == "guzheng":
        return make_guzheng_wave(freq, duration)
    if shape == "dj_scratch":
        return make_dj_scratch_wave(freq, duration)
    return make_basic_wave(freq, duration, shape)


_sound_cache = {}


def get_sound(octave, offset, shape, sustain):
    duration = 3.0 if sustain else 0.9
    key = (octave, offset, shape, sustain)
    snd = _sound_cache.get(key)
    if snd is None:
        freq = freq_for(octave, offset)
        snd = make_wave(freq, duration, shape)
        _sound_cache[key] = snd
    return snd


def play_note(snd, volume, muted):
    channel = snd.play()
    if channel is not None:
        channel.set_volume(0.0 if muted else volume)
    return channel


# ------------------------------------------------------------- drum kit ----

def make_kick(duration=0.35):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 150 * (45.0 / 150.0) ** (t / duration)  # pitch sweep 150Hz -> 45Hz
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    wave = np.sin(phase) * np.exp(-9.0 * t / duration)
    audio = (wave * 0.9 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_snare(duration=0.18):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = np.sin(2 * np.pi * 180 * t) * np.exp(-18.0 * t / duration)
    noise = np.random.uniform(-1, 1, n) * np.exp(-10.0 * t / duration)
    wave = np.clip(0.35 * tone + 0.85 * noise, -1, 1)
    audio = (wave * 0.8 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_hat(open_hat=False):
    duration = 0.28 if open_hat else 0.07
    decay = 6.0 if open_hat else 30.0
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.random.uniform(-1, 1, n) * np.exp(-decay * t / duration)
    audio = (wave * 0.5 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_kick_epic(duration=0.5):
    """Deep, booming war-drum hit for the historical patterns - lower pitch,
    longer sustain and normalized to near-peak so it lands louder/heavier
    than the regular pop/rock kick."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 120 * (40.0 / 120.0) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    wave = np.sin(phase) * np.exp(-4.0 * t / duration)
    wave += 0.25 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-7.0 * t / duration)
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.98 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


def make_snare_epic(duration=0.3):
    """Deep war-drum tom hit for the historical patterns - lower tone, more
    body than the regular snare, normalized louder."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = np.sin(2 * np.pi * 110 * t) * np.exp(-8.0 * t / duration)
    noise = np.random.uniform(-1, 1, n) * np.exp(-14.0 * t / duration)
    wave = 0.6 * tone + 0.6 * noise
    peak = np.max(np.abs(wave)) or 1.0
    audio = (wave / peak * 0.95 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(audio)


_drum_cache = {}


def get_drum_sound(name):
    snd = _drum_cache.get(name)
    if snd is None:
        if name == "kick":
            snd = make_kick()
        elif name == "snare":
            snd = make_snare()
        elif name == "hat":
            snd = make_hat(open_hat=False)
        elif name == "hat_open":
            snd = make_hat(open_hat=True)
        elif name == "kick_epic":
            snd = make_kick_epic()
        elif name == "snare_epic":
            snd = make_snare_epic()
        _drum_cache[name] = snd
    return snd


# 16-step patterns (one bar of 4/4 in 16th notes). hat: 0 off, 1 closed, 2 open.
BEATS = {
    "Rock": {
        "kick":  [1,0,0,0, 0,0,0,0, 1,0,1,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat":   [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
    },
    "Four on the Floor": {
        "kick":  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat":   [0,0,2,0, 0,0,2,0, 0,0,2,0, 0,0,2,0],
    },
    "Hip-Hop": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat":   [1,0,1,1, 0,1,1,0, 1,1,0,1, 1,0,1,1],
    },
    "Funk": {
        "kick":  [1,0,0,1, 0,0,1,0, 0,0,1,0, 0,1,0,0],
        "snare": [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
        "hat":   [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    },
    "Reggae": {
        "kick":  [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "hat":   [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
    },
    "Trap": {
        "kick":  [1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,0],
        "snare": [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "hat":   [1,1,1,1, 1,2,1,1, 1,1,1,1, 1,1,1,2],
    },
    # -- epic / historical war-drum patterns: no hihat where the real ensembles
    # had none, snare doubling as a second drum voice for accents/rolls.
    "Taiko": {
        "kick":  [1,0,0,1, 0,1,0,0, 1,0,0,1, 0,1,1,0],
        "snare": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,1],
        "hat":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "War March": {
        "kick":  [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
        "snare": [0,0,1,1, 0,1,0,1, 0,0,1,1, 0,1,1,1],
        "hat":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "Mongol Gallop": {
        "kick":  [1,0,1,0, 0,1,0,1, 1,0,1,0, 0,1,0,1],
        "snare": [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
        "hat":   [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
    },
    "Viking War Drum": {
        "kick":  [1,0,0,1, 1,0,0,0, 1,0,0,1, 1,0,1,0],
        "snare": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
        "hat":   [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
    },
    "Ottoman Mehter": {
        "kick":  [1,0,0,0, 1,0,1,0, 1,0,0,0, 1,0,1,0],
        "snare": [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
        "hat":   [2,0,0,0, 0,0,0,0, 2,0,0,0, 0,0,0,0],
    },
}
BEAT_NAMES = list(BEATS.keys())
EPIC_BEATS = {"Taiko", "War March", "Mongol Gallop", "Viking War Drum", "Ottoman Mehter"}


SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.json")


def save_recording(events, path=SAVE_PATH):
    with open(path, "w") as f:
        json.dump(events, f)


def load_recording(path=SAVE_PATH):
    with open(path) as f:
        return [tuple(event) for event in json.load(f)]


PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")
PRESET_KEYS = [pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5]

BEAT_SHORT_NAMES = {"Four on the Floor": "4-Floor"}


def short_beat_name(name):
    return BEAT_SHORT_NAMES.get(name, name)


def load_presets(path=PRESETS_PATH):
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}
    return {int(slot): preset for slot, preset in data.items()}


def save_presets(presets, path=PRESETS_PATH):
    with open(path, "w") as f:
        json.dump({str(slot): preset for slot, preset in presets.items()}, f)


# ------------------------------------------------------------------ UI -----

WIDTH, HEIGHT = 900, 440
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Keyboard Piano")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("dejavusans", 48)
font_mid = pygame.font.SysFont("dejavusans", 24)
font_small = pygame.font.SysFont("dejavusans", 18)

BG = (24, 24, 28)
WHITE_KEY = (245, 245, 245)
WHITE_KEY_ACTIVE = (120, 200, 255)
WHITE_KEY_PLAYBACK = (150, 230, 160)
BLACK_KEY = (30, 30, 30)
BLACK_KEY_ACTIVE = (60, 130, 220)
BLACK_KEY_PLAYBACK = (40, 150, 70)
TEXT = (230, 230, 230)
ACCENT = (120, 200, 255)
REC_COLOR = (220, 70, 70)
PLAY_COLOR = (80, 200, 120)
SCRATCH_COLOR = (230, 130, 230)

WHITE_KEYS = [
    (pygame.K_a, "A", 0),
    (pygame.K_s, "S", 2),
    (pygame.K_d, "D", 4),
    (pygame.K_f, "F", 5),
    (pygame.K_g, "G", 7),
    (pygame.K_h, "H", 9),
    (pygame.K_j, "J", 11),
    (pygame.K_k, "K", 12),
]
BLACK_KEYS = [
    (pygame.K_w, "W", 1, 0),
    (pygame.K_e, "E", 3, 1),
    (pygame.K_t, "T", 6, 3),
    (pygame.K_y, "Y", 8, 4),
    (pygame.K_u, "U", 10, 5),
]

KEY_LABEL_ORDER = list(range(13))


KB_X, KB_Y = 40, 255
KB_W, KB_H = WIDTH - 80, 145


def draw_keyboard_frame(recording, playing, now, scratch_flash_expire):
    """Draw a border + badge around the keyboard area showing rec/playback state."""
    margin = 12
    border_rect = pygame.Rect(
        KB_X - margin, KB_Y - margin, KB_W + margin * 2, KB_H + margin * 2
    )

    if now < scratch_flash_expire:
        badge = font_small.render("SCRATCH", True, SCRATCH_COLOR)
        screen.blit(badge, (border_rect.right - badge.get_width() - 6, border_rect.y - 22))

    if recording:
        pulse = 0.5 + 0.5 * abs(math.sin(now * 5))
        thickness = 3 + round(pulse * 3)
        pygame.draw.rect(screen, REC_COLOR, border_rect, thickness, border_radius=10)
        if (now * 2) % 1 < 0.6:  # blinking dot
            pygame.draw.circle(screen, REC_COLOR, (border_rect.x + 10, border_rect.y - 14), 6)
        badge = font_small.render("REC", True, REC_COLOR)
        screen.blit(badge, (border_rect.x + 22, border_rect.y - 22))
    elif playing:
        pulse = 0.6 + 0.4 * abs(math.sin(now * 3))
        color = tuple(int(c * pulse) for c in PLAY_COLOR)
        pygame.draw.rect(screen, color, border_rect, 3, border_radius=10)
        pygame.draw.polygon(
            screen,
            PLAY_COLOR,
            [
                (border_rect.x + 6, border_rect.y - 20),
                (border_rect.x + 6, border_rect.y - 8),
                (border_rect.x + 16, border_rect.y - 14),
            ],
        )
        badge = font_small.render("PLAY", True, PLAY_COLOR)
        screen.blit(badge, (border_rect.x + 22, border_rect.y - 22))


def draw_keyboard(active_keys, playback_keys):
    n_white = len(WHITE_KEYS)
    kb_x, kb_y = KB_X, KB_Y
    kb_w, kb_h = KB_W, KB_H
    white_w = kb_w / n_white

    for i, (key, label, offset) in enumerate(WHITE_KEYS):
        rect = pygame.Rect(kb_x + i * white_w, kb_y, white_w - 2, kb_h)
        if key in active_keys:
            color = WHITE_KEY_ACTIVE
        elif key in playback_keys:
            color = WHITE_KEY_PLAYBACK
        else:
            color = WHITE_KEY
        pygame.draw.rect(screen, color, rect, border_radius=6)
        pygame.draw.rect(screen, (10, 10, 10), rect, 2, border_radius=6)
        lbl = font_small.render(label, True, (20, 20, 20))
        screen.blit(lbl, (rect.centerx - lbl.get_width() / 2, rect.bottom - 28))

    black_w = white_w * 0.6
    black_h = kb_h * 0.6
    for key, label, offset, after_white_idx in BLACK_KEYS:
        x = kb_x + (after_white_idx + 1) * white_w - black_w / 2
        rect = pygame.Rect(x, kb_y, black_w, black_h)
        if key in active_keys:
            color = BLACK_KEY_ACTIVE
        elif key in playback_keys:
            color = BLACK_KEY_PLAYBACK
        else:
            color = BLACK_KEY
        pygame.draw.rect(screen, color, rect, border_radius=4)
        lbl = font_small.render(label, True, (230, 230, 230))
        screen.blit(lbl, (rect.centerx - lbl.get_width() / 2, rect.bottom - 24))


def draw_volume(volume, muted):
    bar_w, bar_h = 120, 14
    x, y = WIDTH - 40 - bar_w, 28
    pygame.draw.rect(screen, (60, 60, 65), (x, y, bar_w, bar_h), border_radius=4)
    fill_w = 0 if muted else int(bar_w * volume)
    fill_color = (200, 90, 90) if muted else ACCENT
    if fill_w > 0:
        pygame.draw.rect(screen, fill_color, (x, y, fill_w, bar_h), border_radius=4)
    pygame.draw.rect(screen, (10, 10, 10), (x, y, bar_w, bar_h), 2, border_radius=4)
    icon = "MUTE" if muted else "VOL"
    label = font_small.render(icon, True, (150, 150, 150))
    screen.blit(label, (x - label.get_width() - 10, y - 2))


BUTTON_W, BUTTON_H, BUTTON_GAP = 76, 26, 8
SAVE_RECT = pygame.Rect(
    WIDTH - 40 - BUTTON_W * 2 - BUTTON_GAP, 52, BUTTON_W, BUTTON_H
)
LOAD_RECT = pygame.Rect(WIDTH - 40 - BUTTON_W, 52, BUTTON_W, BUTTON_H)


def draw_button(rect, label_text, enabled):
    bg = (50, 50, 58) if enabled else (35, 35, 38)
    text_color = TEXT if enabled else (90, 90, 90)
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    pygame.draw.rect(screen, (10, 10, 10), rect, 2, border_radius=6)
    lbl = font_small.render(label_text, True, text_color)
    screen.blit(lbl, (rect.centerx - lbl.get_width() / 2, rect.centery - lbl.get_height() / 2))


def draw_save_load(has_recording, has_saved_file, message):
    draw_button(SAVE_RECT, "SAVE", has_recording)
    draw_button(LOAD_RECT, "LOAD", has_saved_file)
    if message:
        msg_surf = font_small.render(message, True, (150, 150, 150))
        screen.blit(msg_surf, (LOAD_RECT.right - msg_surf.get_width(), 82))


BEAT_STEP_COLOR = (55, 55, 62)
BEAT_KICK_COLOR = (220, 90, 90)
BEAT_SNARE_COLOR = (230, 170, 70)
BEAT_HAT_COLOR = (90, 170, 220)
BEAT_CURRENT_BORDER = (240, 240, 240)


def draw_beat_bar(pattern, current_step, beat_on):
    n = 16
    x0, y0 = KB_X, 205
    w, h = KB_W, 16
    step_w = w / n
    for i in range(n):
        rect = pygame.Rect(x0 + i * step_w, y0, step_w - 3, h)
        if pattern["kick"][i]:
            color = BEAT_KICK_COLOR
        elif pattern["snare"][i]:
            color = BEAT_SNARE_COLOR
        elif pattern["hat"][i]:
            color = BEAT_HAT_COLOR
        else:
            color = BEAT_STEP_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=3)
        if beat_on and i == current_step:
            pygame.draw.rect(screen, BEAT_CURRENT_BORDER, rect, 2, border_radius=3)


def draw_presets(presets, message):
    y = 227
    if message:
        text, color = message, ACCENT
    elif presets:
        entries = [
            f"F{slot + 1}:{preset.get('instrument', '?')}/{short_beat_name(preset.get('beat', '?'))}"
            for slot, preset in sorted(presets.items())
        ]
        text = "Presets: " + "  ".join(entries) + "   (Ctrl+F save, F load)"
        color = (150, 150, 150)
    else:
        text = "Presets: Ctrl+F1-F5 saves the current instrument+beat combo, F1-F5 loads it"
        color = (150, 150, 150)
    surf = font_small.render(text, True, color)
    screen.blit(surf, (KB_X, y))


def main():
    octave = 4
    waveform = "sine"
    sustain = False
    volume = 0.8
    muted = False
    active_keys = set()
    last_note_text = ""
    recording = False
    playing = False
    record_start = 0.0
    recorded_events = []  # list of (t_offset, octave, offset, shape)
    loaded_recording = []
    playback_keys = {}  # key -> time.time() when highlight should clear
    message = ""
    message_expire = 0.0

    beat_on = False
    beat_index = 0
    bpm = 100
    beat_start = 0.0
    last_beat_step = -1
    beat_display_step = 0

    presets = load_presets()
    preset_message = ""
    preset_message_expire = 0.0

    scratch_flash_expire = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_z:
                    octave = max(0, octave - 1)
                elif event.key == pygame.K_x:
                    octave = min(8, octave + 1)

                elif event.key == pygame.K_SPACE:
                    sustain = True

                elif event.key in WAVEFORMS:
                    waveform = WAVEFORMS[event.key]

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    volume = max(0.0, round(volume - 0.1, 2))
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    volume = min(1.0, round(volume + 0.1, 2))
                elif event.key == pygame.K_m:
                    muted = not muted

                elif event.key == pygame.K_r:
                    if not recording:
                        recording = True
                        recorded_events = []
                        record_start = time.time()
                    else:
                        recording = False

                elif event.key == pygame.K_p and recorded_events:
                    loaded_recording = list(recorded_events)
                    playing = True
                    play_start = time.time()
                    play_index = 0

                elif event.key == pygame.K_b:
                    beat_on = not beat_on
                    if beat_on:
                        beat_start = time.time()
                        last_beat_step = -1

                elif event.key == pygame.K_n:
                    beat_index = (beat_index + 1) % len(BEAT_NAMES)
                    last_beat_step = -1

                elif event.key == pygame.K_LEFTBRACKET:
                    bpm = max(40, bpm - 5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    bpm = min(240, bpm + 5)

                elif event.key in PRESET_KEYS:
                    slot = PRESET_KEYS.index(event.key)
                    if event.mod & pygame.KMOD_CTRL:
                        presets[slot] = {
                            "instrument": waveform,
                            "beat": BEAT_NAMES[beat_index],
                            "bpm": bpm,
                        }
                        save_presets(presets)
                        preset_message = f"Saved preset F{slot + 1}"
                    else:
                        preset = presets.get(slot)
                        if preset and preset.get("beat") in BEATS:
                            waveform = preset.get("instrument", waveform)
                            beat_index = BEAT_NAMES.index(preset["beat"])
                            bpm = preset.get("bpm", bpm)
                            beat_on = True
                            beat_start = time.time()
                            last_beat_step = -1
                            preset_message = f"Loaded preset F{slot + 1}"
                        else:
                            preset_message = f"Preset F{slot + 1} is empty"
                    preset_message_expire = time.time() + 2.0

                elif event.key == pygame.K_SLASH:
                    scratch_snd = get_sound(octave, 0, "dj_scratch", False)
                    play_note(scratch_snd, volume, muted)
                    last_note_text = note_name(octave, 0) + " (scratch)"
                    scratch_flash_expire = time.time() + 0.3
                    if recording:
                        recorded_events.append(
                            (time.time() - record_start, octave, 0, "dj_scratch")
                        )

                elif event.key in KEY_OFFSETS and event.key not in active_keys:
                    active_keys.add(event.key)
                    offset = KEY_OFFSETS[event.key]
                    snd = get_sound(octave, offset, waveform, sustain)
                    play_note(snd, volume, muted)
                    last_note_text = note_name(octave, offset)
                    if recording:
                        recorded_events.append(
                            (time.time() - record_start, octave, offset, waveform)
                        )

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    sustain = False
                if event.key in active_keys:
                    active_keys.discard(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if SAVE_RECT.collidepoint(event.pos):
                    if recorded_events:
                        save_recording(recorded_events)
                        message = f"Saved {len(recorded_events)} notes"
                    else:
                        message = "Nothing to save"
                    message_expire = time.time() + 2.0

                elif LOAD_RECT.collidepoint(event.pos):
                    try:
                        recorded_events = load_recording()
                    except FileNotFoundError:
                        message = "No saved recording found"
                    except (json.JSONDecodeError, ValueError):
                        message = "Saved recording is corrupt"
                    else:
                        message = f"Loaded {len(recorded_events)} notes"
                        loaded_recording = list(recorded_events)
                        playing = True
                        play_start = time.time()
                        play_index = 0
                    message_expire = time.time() + 2.0

        now = time.time()

        # drive recorded playback without blocking the event loop
        if playing:
            elapsed = now - play_start
            while play_index < len(loaded_recording) and loaded_recording[play_index][0] <= elapsed:
                _, o, off, shape = loaded_recording[play_index]
                play_note(get_sound(o, off, shape, False), volume, muted)
                last_note_text = note_name(o, off) + " (playback)"
                key = REVERSE_KEY_OFFSETS.get(off)
                if key is not None:
                    playback_keys[key] = now + 0.25
                play_index += 1
            if play_index >= len(loaded_recording):
                playing = False

        playback_keys = {k: t for k, t in playback_keys.items() if t > now}
        if now > message_expire:
            message = ""
        if now > preset_message_expire:
            preset_message = ""

        if beat_on:
            step_dur = 60.0 / bpm / 4.0
            total_steps = int((now - beat_start) / step_dur)
            if total_steps != last_beat_step:
                last_beat_step = total_steps
                cur = total_steps % 16
                beat_display_step = cur
                beat_name = BEAT_NAMES[beat_index]
                pattern = BEATS[beat_name]
                is_epic = beat_name in EPIC_BEATS
                if pattern["kick"][cur]:
                    play_note(get_drum_sound("kick_epic" if is_epic else "kick"), volume, muted)
                if pattern["snare"][cur]:
                    play_note(get_drum_sound("snare_epic" if is_epic else "snare"), volume, muted)
                hat_val = pattern["hat"][cur]
                if hat_val == 1:
                    play_note(get_drum_sound("hat"), volume, muted)
                elif hat_val == 2:
                    play_note(get_drum_sound("hat_open"), volume, muted)

        screen.fill(BG)

        title = font_mid.render("Keyboard Piano", True, ACCENT)
        screen.blit(title, (40, 20))

        note_surf = font_big.render(last_note_text or "-", True, TEXT)
        screen.blit(note_surf, (40, 60))

        vol_text = "MUTED" if muted else f"{int(volume * 100)}%"
        status = (
            f"Octave: {octave}   Instrument: {INSTRUMENT_LABELS.get(waveform, waveform)}   "
            f"Sustain: {'ON' if sustain else 'off'}   "
            f"Vol: {vol_text}   "
            f"Rec: {'ON' if recording else 'off'}   "
            f"Beat: {BEAT_NAMES[beat_index]} {bpm}bpm {'ON' if beat_on else 'off'}"
        )
        status_surf = font_small.render(status, True, TEXT)
        screen.blit(status_surf, (40, 130))

        help_line1 = font_small.render(
            "A S D F G H J K = C D E F G A B C   |   W E T Y U = sharps",
            True,
            (150, 150, 150),
        )
        help_line2 = font_small.render(
            "Z/X octave  |  Space sustain  |  1-9/0/,/. instrument  |  -/= volume  |  M mute  |  "
            "R record  |  P play  |  B beat  |  N next beat  |  [/] tempo  |  / scratch  |  Esc quit",
            True,
            (150, 150, 150),
        )
        screen.blit(help_line1, (40, 158))
        screen.blit(help_line2, (40, 180))

        draw_volume(volume, muted)
        draw_save_load(bool(recorded_events), os.path.exists(SAVE_PATH), message)
        draw_beat_bar(BEATS[BEAT_NAMES[beat_index]], beat_display_step, beat_on)
        draw_presets(presets, preset_message)
        draw_keyboard_frame(recording, playing, now, scratch_flash_expire)
        draw_keyboard(active_keys, playback_keys)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
