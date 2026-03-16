#!/bin/bash
# AVH RAG Pipeline Setup Script

set -e  # Exit on error

echo "🚀 Setting up AVH RAG Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (for A40)
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create project directories
echo "📁 Creating project structure..."
mkdir -p data/tei_files
mkdir -p data/vector_db
mkdir -p models/embeddings
mkdir -p models/llm
mkdir -p outputs
mkdir -p notebooks
mkdir -p src

# Create .env file template
cat > .env << EOF
# AVH RAG Pipeline Configuration

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=avh_documents

# Models
EMBEDDING_MODEL=intfloat/multilingual-e5-large
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Paths
TEI_FILES_PATH=./data/tei_files
VECTOR_DB_PATH=./data/vector_db

# Generation Settings
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_K=5

EOF

echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Start Qdrant: docker-compose up -d"
echo "3. Place your TEI files in: data/tei_files/"
echo "4. Run the pipeline: python src/main.py"