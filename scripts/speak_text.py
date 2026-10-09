"""
CLI tool to speak a text phrase using the TextToSpeech service.

Usage:
    python scripts/speak_text.py <text>

Example:
    python scripts/speak_text.py "Hola mundo"
"""

import argparse
import sys

from signvision.services import TextToSpeech


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak a text phrase")
    parser.add_argument("text", type=str, help="Text to speak")

    args = parser.parse_args()

    text_to_speech = TextToSpeech()

    try:
        text_to_speech.speak(args.text)
    finally:
        text_to_speech.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
