import os
import sys

# Must be set before `piano` (and therefore pygame) is imported anywhere,
# so tests run headless without a real display or audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
