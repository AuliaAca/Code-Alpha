# CodeAlpha

This repository contains four AI-based tasks from the CodeAlpha internship program, combined into one project with a shared demo flow and a single web interface.

## Project overview

The project includes four independent modules:

1. [Task1](Task1/) — language translation tool
2. [Task2](Task2/) — FAQ-based chatbot
3. [Task3](Task3/) — AI music generation using an LSTM model
4. [Task4](Task4/) — object detection and tracking from video or webcam

These modules are also connected through the shared hub system in [hub/](hub/), which allows the project to act like a single integrated AI demo.

## Features

- Text translation with fallback providers
- FAQ matching using NLP and similarity scoring
- Music generation based on training data
- Object detection and tracking pipeline
- Streamlit-based user interface
- CLI support for individual tasks

## Tech stack

- Python 3.10+
- Streamlit
- PyTorch
- OpenCV
- scikit-learn
- NumPy
- requests
- NLTK
- Ultralytics YOLO (optional for detection)
- pytest

## Repository structure

```text
CodeAlpha/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── common/
├── hub/
├── Task1/
├── Task2/
├── Task3/
├── Task4/
├── docs/
├── scripts/
├── hub_tests/
└── LICENSE
```

## Setup

### 1) Clone the repository

```bash
git clone https://github.com/AuliaAca/Code-Alpha.git
cd Code-Alpha
```

### 2) Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the project

#### Web app

```bash
streamlit run app.py
```

#### Individual tasks

```bash
python Task1/cli.py "good morning" -t id
python Task2/cli.py
python Task3/cli.py generate --length 64 --temperature 0.9
python Task4/cli.py --demo --show
```

## Environment variables

This project uses environment-based configuration for API keys, especially for translation services. Sensitive values should not be committed.

- Copy `.env.example` to `.env`
- Add your API keys there
- Keep `.env` local and untracked

The repository already includes `.env` in `.gitignore`, so it will not be pushed.

## Important notes

- Do not commit `.env`, `.venv/`, generated outputs, or downloaded model files.
- The project includes large dependencies and runtime artifacts that should remain local.
- If you want to use external translation APIs, configure the keys in `.env` before running the relevant task.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. Please keep changes focused, keep the code readable, and avoid committing secrets or generated local artifacts.

