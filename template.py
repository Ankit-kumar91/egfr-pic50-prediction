import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "EGFR_drug_discovery_pipeline"

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_loader.py",
    "src/components/feature_engineering.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluator.py",
    "src/components/model_moniterring.py",
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils/__init__.py",
    "src/utils/model_utils.py",
    "configs/config.yaml",
    "setup.py", 
    "main.py",
    "app.py"
    ]

list_of_dirs =[
    "data",
    "structure",
    "notebooks",
    "models",
    "results"
 ]


# Create directories 
for dir_path in list_of_dirs:
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Directory created: {dir_path}")

# Create files
for file in list_of_files:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    if not file.exists() or file.stat().st_size == 0:
        file.touch()
        logging.info(f"Created file: {file}")
    else:
        logging.info(f"File already exists: {file}")

