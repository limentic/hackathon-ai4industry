# FastAPI Application

A simple FastAPI application.

## Prerequisites

- [Python 3.9](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)

## Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/limentic/hackathon-ai4industry
   cd hackathon-ai4industry
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Launching the Application

1. **Run the FastAPI application** using Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```
   You should see the app running.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.