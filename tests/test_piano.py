import json

import numpy as np
import pytest

import piano


# --------------------------------------------------------------- pitch math

def test_midi_number_middle_c():
    assert piano.midi_number(4, 0) == 60


def test_freq_for_a4_is_440hz():
    assert piano.freq_for(4, 9) == pytest.approx(440.0, abs=0.01)


def test_freq_for_middle_c():
    assert piano.freq_for(4, 0) == pytest.approx(261.63, abs=0.01)


def test_note_name_basic():
    assert piano.note_name(4, 0) == "C4"
    assert piano.note_name(4, 1) == "C#4"
    assert piano.note_name(4, 11) == "B4"


def test_note_name_wraps_to_next_octave():
    # offset 12 is the octave's C, one octave up
    assert piano.note_name(4, 12) == "C5"


def test_reverse_key_offsets_round_trip():
    for key, offset in piano.KEY_OFFSETS.items():
        assert piano.REVERSE_KEY_OFFSETS[offset] == key


# ------------------------------------------------------------- instruments

def test_every_waveform_has_a_label():
    for shape in piano.WAVEFORMS.values():
        assert shape in piano.INSTRUMENT_LABELS, f"{shape} missing from INSTRUMENT_LABELS"


@pytest.mark.parametrize("shape", sorted(set(piano.WAVEFORMS.values())))
def test_make_wave_produces_sound_for_every_instrument(shape):
    snd = piano.make_wave(piano.freq_for(4, 0), 0.2, shape)
    assert snd is not None
    assert snd.get_length() > 0


def test_make_wave_unknown_shape_falls_back_to_basic_sine():
    snd = piano.make_wave(piano.freq_for(4, 0), 0.2, "not_a_real_instrument")
    assert snd is not None


def test_karplus_strong_core_output_shape_and_bounds():
    out = piano.karplus_strong_core(220.0, 0.1)
    assert isinstance(out, np.ndarray)
    assert len(out) == int(piano.SAMPLE_RATE * 0.1)
    assert np.all(np.isfinite(out))


def test_get_sound_is_cached():
    snd1 = piano.get_sound(4, 0, "sine", False)
    snd2 = piano.get_sound(4, 0, "sine", False)
    assert snd1 is snd2


def test_get_sound_sustain_uses_longer_duration():
    short = piano.get_sound(5, 0, "sine", False)
    long = piano.get_sound(5, 0, "sine", True)
    assert long.get_length() > short.get_length()


# ------------------------------------------------------------------- drums

@pytest.mark.parametrize(
    "name", ["kick", "snare", "hat", "hat_open", "kick_epic", "snare_epic"]
)
def test_get_drum_sound_returns_valid_sound(name):
    snd = piano.get_drum_sound(name)
    assert snd is not None
    assert snd.get_length() > 0


def test_epic_drum_kit_is_louder_than_regular_kit():
    def peak(snd):
        return int(np.max(np.abs(piano.pygame.sndarray.array(snd))))

    assert peak(piano.get_drum_sound("kick_epic")) > peak(piano.get_drum_sound("kick"))
    assert peak(piano.get_drum_sound("snare_epic")) > peak(piano.get_drum_sound("snare"))


# ------------------------------------------------------------------- beats

def test_beat_names_matches_beats_dict():
    assert piano.BEAT_NAMES == list(piano.BEATS.keys())


@pytest.mark.parametrize("name", piano.BEAT_NAMES)
def test_beat_pattern_shape_is_valid(name):
    pattern = piano.BEATS[name]
    for voice in ("kick", "snare", "hat"):
        assert len(pattern[voice]) == 16
    assert all(v in (0, 1) for v in pattern["kick"])
    assert all(v in (0, 1) for v in pattern["snare"])
    assert all(v in (0, 1, 2) for v in pattern["hat"])


def test_epic_beats_is_subset_of_beat_names():
    assert piano.EPIC_BEATS <= set(piano.BEAT_NAMES)


def test_short_beat_name_abbreviates_known_long_name():
    assert piano.short_beat_name("Four on the Floor") == "4-Floor"


def test_short_beat_name_passes_through_unknown_name():
    assert piano.short_beat_name("Rock") == "Rock"


# -------------------------------------------------------- persistence i/o

def test_save_load_recording_round_trip(tmp_path):
    path = tmp_path / "recording.json"
    events = [(0.0, 4, 0, "sine"), (0.5, 4, 4, "guitar")]
    piano.save_recording(events, path=str(path))
    loaded = piano.load_recording(path=str(path))
    assert loaded == events


def test_save_load_presets_round_trip(tmp_path):
    path = tmp_path / "presets.json"
    presets = {0: {"instrument": "guzheng", "beat": "Reggae", "bpm": 110}}
    piano.save_presets(presets, path=str(path))
    loaded = piano.load_presets(path=str(path))
    assert loaded == presets


def test_load_presets_missing_file_returns_empty_dict(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert piano.load_presets(path=str(path)) == {}


def test_load_presets_corrupt_file_returns_empty_dict(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not valid json{{{")
    assert piano.load_presets(path=str(path)) == {}


def test_presets_json_round_trips_through_actual_json(tmp_path):
    path = tmp_path / "presets.json"
    presets = {1: {"instrument": "pipa", "beat": "Trap", "bpm": 90}}
    piano.save_presets(presets, path=str(path))
    with open(path) as f:
        raw = json.load(f)
    assert raw == {"1": {"instrument": "pipa", "beat": "Trap", "bpm": 90}}
