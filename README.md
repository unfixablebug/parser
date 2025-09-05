# parser
### Install System dependencies:
```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev python3.pip
python3 -m pip install --upgrade pip setuptools wheel
sudo apt install python3.12-venv
```

### Create A Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # on Linux/Mac
.\.venv\Scripts\activate    # on Windows
```
### Install Python dependencies
```bash
pip install -r requirements.txt
```
