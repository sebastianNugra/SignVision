# SignVision AI

Desktop application for real-time sign language translation using computer vision and artificial intelligence.

## Technologies

| Technology | Purpose |
|---|---|
| Python 3.12+ | Core language |
| OpenCV | Video capture and image processing |
| MediaPipe | Hand landmark detection |
| NumPy | Numerical computation |
| TensorFlow | AI model inference |
| SQLite | Local data persistence |
| CustomTkinter | Desktop GUI |
| Pillow | Image handling |
| pyttsx3 | Text-to-speech |

## Architecture

```
src/signvision/
├── camera/        # Webcam capture and frame management
├── vision/        # Hand detection and landmark extraction
├── models/        # AI model loading and gesture classification
├── services/      # Business logic orchestration
├── database/      # SQLite persistence layer
├── gui/           # Desktop interface (CustomTkinter)
├── config/        # Application settings and paths
└── utils/         # Shared helpers, logging, exceptions
```

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git

### Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black ruff mypy
```

### Run the Application

```bash
python src/main.py
```

### Run Tests

```bash
pytest
```

### Code Quality

```bash
black .
ruff check .
mypy src/
```

## Project Structure

```
signvision-ai/
├── src/
│   └── signvision/
│       ├── __init__.py
│       ├── main.py              # Application entry point
│       ├── camera/              # Webcam management
│       │   ├── __init__.py
│       │   ├── camera_manager.py
│       │   └── frame_reader.py
│       ├── vision/              # Hand detection pipeline
│       │   ├── __init__.py
│       │   ├── hand_detector.py
│       │   ├── landmark_extractor.py
│       │   └── preprocessor.py
│       ├── models/              # AI classification
│       │   ├── __init__.py
│       │   ├── gesture_classifier.py
│       │   ├── model_loader.py
│       │   └── label_map.py
│       ├── services/            # Business logic
│       │   ├── __init__.py
│       │   ├── translation_service.py
│       │   └── text_to_speech.py
│       ├── database/            # Persistence layer
│       │   ├── __init__.py
│       │   ├── db_connection.py
│       │   ├── db_models.py
│       │   └── repositories.py
│       ├── gui/                 # User interface
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   ├── camera_view.py
│       │   ├── results_panel.py
│       │   └── components.py
│       ├── config/              # Settings & paths
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   └── paths.py
│       └── utils/               # Helpers & utilities
│           ├── __init__.py
│           ├── helpers.py
│           ├── logger.py
│           └── exceptions.py
├── tests/
│   ├── unit/
│   └── integration/
├── assets/
│   ├── icons/
│   ├── images/
│   └── sounds/
├── trained_models/              # Saved AI models
├── docs/
├── scripts/
├── pyproject.toml
├── requirements.txt
├── .editorconfig
├── .gitignore
├── LICENSE
└── README.md
```

## Roadmap

- [ ] Project setup and configuration
- [ ] Camera module implementation
- [ ] Hand detection with MediaPipe
- [ ] Landmark extraction and normalization
- [ ] AI model training pipeline
- [ ] Gesture classification
- [ ] Translation service integration
- [ ] Text-to-speech output
- [ ] GUI development with CustomTkinter
- [ ] SQLite database for history
- [ ] Unit and integration tests
- [ ] Performance optimization
- [ ] Packaging and distribution

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
