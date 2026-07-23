# Credit Risk Intelligence System (`credit-risk-agent`)

A hybrid credit risk assessment and automated underwriting system combining a PyTorch deep learning model for default probability prediction with a GigaChat ReAct AI Agent for credit scoring analysis.

---

## Key Features

- **ReAct AI Agent**: Conducts automated multi-step credit risk analysis using function calling and outputs structured underwriting reports.
- **Real-Time Streaming**: Streams agent reasoning steps (`thought`), tool execution (`tool_call`), observations (`observation`), and final decisions (`final`).
- **Hybrid ML Model (`CreditDefaultPredictor`)**: PyTorch neural network processing 6-month historical payment sequences alongside static demographic features.
- **Relational Data Storage**: SQLite database for client profiles (`clients`), payment history (`payment_history`), and ground truth labels (`ground_truth`).
- **Streamlit Web Application**: Interactive client profiling dashboard and real-time agent chat interface.
- **CLI Suite**: Command-line interface for batch evaluation, interactive chat mode, and model evaluation.

---

## Tech Stack

- **Core**: Python 3.12+, Poetry
- **Machine Learning**: PyTorch, Scikit-Learn, Pandas, NumPy
- **LLM & Agent**: GigaChat API SDK, ReAct Pattern
- **Web UI**: Streamlit
- **Database**: SQLite3
- **Quality & Testing**: Pytest, Ruff, Mypy, Pre-commit

---

## Installation & Setup

### 1. Prerequisites
- Python `>= 3.12`
- Poetry

### 2. Installation
```bash
git clone https://github.com/iraedeus/credit-risk-agent.git
cd credit-risk-agent
poetry install
```

### 3. Environment Configuration
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Configure credentials in `.env`:
```ini
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
GIGACHAT_CREDENTIALS=your_gigachat_authorization_data
```

---

## Data Pipeline & Model Training

### 1. Download & Prepare Dataset
Download the UCI Credit Card dataset from Kaggle and populate the SQLite database:
```bash
poetry run download-dataset
```

### 2. Train Model
Train the `CreditDefaultPredictor` PyTorch model (artifacts saved to `artifacts/model.pt` and `artifacts/scaler.json`):
```bash
poetry run train-model
```

Evaluate model performance on the test split:
```bash
poetry run train-model --view-quality
```

---

## Usage

### 1. Web Application (Streamlit)
```bash
poetry run streamlit run credit_risk_agent/app/main.py
```
Open `http://localhost:8501` to access:
- **Client Profile**: Financial metrics, utilization trends, and payment discipline.
- **AI Agent Chat**: Interactive chat interface with real-time reasoning visualization.

### 2. Command Line Interface (CLI)

List available test client IDs:
```bash
poetry run credit-risk-agent --list-clients
```

View financial info for a specific client:
```bash
poetry run credit-risk-agent --get-client-info -c 100
```

Run single prompt assessment:
```bash
poetry run credit-risk-agent --prompt "Evaluate credit risk for client 105" --verbose
```

Interactive terminal chat mode:
```bash
poetry run credit-risk-agent --chat --verbose
```

---

## Testing & Quality

Run test suite:
```bash
poetry run pytest
```

Code linting and formatting:
```bash
poetry run ruff check .
```

Type checking:
```bash
poetry run mypy credit_risk_agent
```

Pre-commit validation:
```bash
poetry run pre-commit run --all-files
```

---

## Repository Structure

```
credit-risk-agent/
├── artifacts/              # Model weights (model.pt) and scaler (scaler.json)
├── credit_risk_agent/      # Main package source code
│   ├── agent/              # ReAct Agent logic, GigaChat API integration, tools, events
│   ├── app/                # Streamlit UI pages and CLI entrypoint
│   ├── data/               # Preprocessing pipelines and StandardScaler
│   ├── model/              # PyTorch model definitions and PyTorch Dataset
│   └── config.py           # Paths and hyperparameter configuration
├── data/                   # SQLite database files (database.db, train/test split DBs)
├── docs/                   # Documentation and ER diagrams
├── notebooks/              # Data analysis and model exploration notebooks
├── scripts/                # Data download and training scripts
├── tests/                  # Pytest unit and integration test suite
├── pyproject.toml          # Poetry dependencies and tool configurations
└── README.md               # Project documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
