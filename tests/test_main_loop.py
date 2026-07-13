"""Drives the real main() event loop with synthetic pygame events - the same
technique used manually throughout development to verify features live.
Covers the interactive loop's branches without needing to refactor main()
into smaller testable pieces."""

import threading
import time

import pygame

import piano


def post(key, down=True, mod=0):
    ev = pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=key, mod=mod, unicode="")
    pygame.event.post(ev)


def tap(key, mod=0):
    post(key, True, mod)
    post(key, False)


def click(rect):
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))


def _redirect_persistence(monkeypatch, tmp_path):
    """Point save/load default paths at a scratch dir so the test suite
    never touches a real recording.json/presets.json on disk."""
    rec_path = str(tmp_path / "recording.json")
    preset_path = str(tmp_path / "presets.json")
    monkeypatch.setattr(piano.save_recording, "__defaults__", (rec_path,))
    monkeypatch.setattr(piano.load_recording, "__defaults__", (rec_path,))
    monkeypatch.setattr(piano.save_presets, "__defaults__", (preset_path,))
    monkeypatch.setattr(piano.load_presets, "__defaults__", (preset_path,))


def test_main_loop_exercises_full_event_handling(monkeypatch, tmp_path):
    _redirect_persistence(monkeypatch, tmp_path)
    # main() calls pygame.quit() on exit, which would tear down the shared
    # mixer/display for every other test in this process - no-op it here.
    monkeypatch.setattr(piano.pygame, "quit", lambda: None)

    def driver():
        time.sleep(0.15)

        # octave shift
        tap(pygame.K_z)
        tap(pygame.K_x)

        # sustain held while switching through every instrument
        post(pygame.K_SPACE, True)
        for key in piano.WAVEFORMS:
            tap(key)
        post(pygame.K_SPACE, False)

        # volume / mute
        tap(pygame.K_MINUS)
        tap(pygame.K_EQUALS)
        tap(pygame.K_m)
        tap(pygame.K_m)  # unmute again

        # record a note, stop, play it back
        tap(pygame.K_r)
        time.sleep(0.02)
        post(pygame.K_a, True)
        time.sleep(0.02)
        post(pygame.K_a, False)
        time.sleep(0.02)
        tap(pygame.K_r)
        time.sleep(0.02)
        tap(pygame.K_p)
        time.sleep(0.05)

        # SAVE then LOAD via the on-screen buttons (redirected paths)
        click(piano.SAVE_RECT)
        time.sleep(0.05)
        click(piano.LOAD_RECT)
        time.sleep(0.05)

        # backing beat: on, cycle every pattern, adjust tempo, let it tick
        tap(pygame.K_b)
        for _ in range(len(piano.BEAT_NAMES)):
            tap(pygame.K_n)
        tap(pygame.K_LEFTBRACKET)
        tap(pygame.K_RIGHTBRACKET)
        time.sleep(0.3)

        # presets: save a slot, load it back, then try an empty slot
        post(pygame.K_F1, True, mod=pygame.KMOD_CTRL)
        post(pygame.K_F1, False)
        time.sleep(0.02)
        tap(pygame.K_F1)
        time.sleep(0.02)
        tap(pygame.K_F2)

        tap(pygame.K_b)  # beat off

        # scratch stab, once while recording to hit that branch too
        tap(pygame.K_r)
        time.sleep(0.02)
        tap(pygame.K_SLASH)
        time.sleep(0.02)
        tap(pygame.K_r)

        time.sleep(0.1)
        post(pygame.K_ESCAPE, True)

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    piano.main()
    t.join(timeout=2)
    assert not t.is_alive()


def test_main_loop_handles_quit_event(monkeypatch):
    monkeypatch.setattr(piano.pygame, "quit", lambda: None)

    def driver():
        time.sleep(0.1)
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    piano.main()
    t.join(timeout=2)
    assert not t.is_alive()
