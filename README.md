# Resume Information Extraction Framework

A comprehensive framework for extracting structured information from resumes and evaluating different Large Language Models (LLMs) on information extraction tasks.

## Project Overview

This project provides a complete pipeline for:

1. Extracting structured information from resume PDFs
2. Utilizing multiple LLM models for information extraction
3. Evaluating model performance with metrics
4. Visualizing results for comparison

## Project Structure

```
.
├── app.py                   # Flask web application
├── evaluation.py            # Core evaluation logic
├── view_evaluation.py       # Report generation
├── pdf_processing.py        # PDF processing utilities
├── llm_integration.py       # LLM API connections
├── ground_truth/            # Ground truth data
│   └── ground_truth.json    # Ground truth information
├── results/                 # Model results
├── evaluation_charts/       # Generated evaluation charts
├── templates/               # HTML templates
├── static/                  # Static assets
│   ├── css/                 # CSS files
│   └── images/              # Image assets
├── uploads/                 # Uploaded PDF files
├── run.sh                   # Script to run the application
├── Dockerfile               # Docker configuration file
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using run.sh (Recommended)

The easiest way to run the application is using the provided `run.sh` script:

```bash
# Make the script executable (Linux/Mac)
chmod +x run.sh

# Run the web application
./run.sh
```

For Windows users using PowerShell:

```powershell
# Run the web application
bash run.sh

# Alternative without bash
python app.py
```

### Option 2: Running Components Individually

You can also run the components individually:

```bash
# Run the web application
python app.py

# Run model evaluation
python evaluation.py

# Generate evaluation report
python view_evaluation.py
```

## Docker Deployment

This project can be easily deployed using Docker.

### Prerequisites

- Docker
- Docker Compose

### Building and Running with Docker Compose

```bash
# Build and start the containers
docker-compose up -d

# The application will be available at http://localhost:5000
```

### Running Ollama Models

When using Docker, the application is configured to connect to Ollama running in a separate container. You'll need to pull the required models:

```bash
# Connect to the Ollama container
docker exec -it ollama bash

# Pull the required models
ollama pull llama3
ollama pull mistral
ollama pull phi

# Exit the container
exit
```

### Stopping the Containers

```bash
docker-compose down
```

### Using Docker Without Compose

If you prefer to run just the application container:

```bash
# Build the Docker image
docker build -t cv-extractor .

# Run the container
docker run -p 5000:5000 -v ./uploads:/app/uploads -v ./results:/app/results cv-extractor
```

Note: In this case, you'll need to update the Ollama API URL in the code to point to your Ollama installation.

## Command-line Arguments

Note: The current version doesn't accept command-line arguments like 'app', 'evaluate', or 'report'. If you try to use these (e.g., `python clean_main.py web`), you'll get an error. The correct usage is shown above.

## Web Application

The web application provides an interface to:
- Upload resume PDFs
- Process them with different LLM models (LLaMA 3, Mistral, Phi-2)
- View the extracted information in a structured format
- Enable OCR for image-based PDFs

## Model Evaluation

The evaluation functionality:
- Compares model outputs against ground truth data
- Calculates precision, recall, and F1 scores
- Generates performance charts and visualizations
- Saves results to `evaluation_results.json` and charts to the `evaluation_charts/` directory

## Supported Models

The framework supports the following LLM models:
- Phi-2
- LLaMA 3
- Mistral

## Evaluation Metrics

Models are evaluated using:
- Precision
- Recall
- F1 Score

Metrics are calculated for each information field (name, email, education, etc.) and aggregated for overall performance.

## Adding Custom Models

To add custom models:
1. Create an integration in `llm_integration.py`
2. Update model patterns in `evaluation.py`

## Troubleshooting

If you encounter errors:

1. Ensure Ollama is running locally with the required models
2. Check the `uploads`, `results`, and `ground_truth` directories exist
3. Verify that `ground_truth.json` is properly formatted
4. Make sure you're using the correct commands as detailed in the Usage section

## License

MIT License 