import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html",
]

for file_path_str in list_of_files:
    filepath = Path(file_path_str)
    directory = filepath.parent

    try:
        # Create directory if it does not exist and is not current directory
        if directory != Path('.'):
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {directory}")

        # Create empty file if missing or zero length
        if not filepath.exists() or filepath.stat().st_size == 0:
            filepath.touch(exist_ok=True)
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"File already exists and is non-empty: {filepath}")

    except Exception as e:
        logging.error(f"Failed processing {filepath}: {e}")