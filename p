[project]
name = "no-sql-kingdom"
version = "0.1.0"
description = "A project about all kind of NoSQL databases"
requires-python = ">=3.11,<3.13"
readme = "README.md"
dependencies = [
    "pymongo==4.13.2",       
    "pydantic==2.11.7",      
    "python-dotenv==1.1.1",
    "neo4j==5.28.2",
    "redis==6.3.0",
    "gradio==5.41.0",
    "ipykernel"
    # or if you want -- streamlit
    # "streamlit==1.48.0"
]

[tool.ruff]
line-length = 90
lint.select = ["E", "F", "W", "C90", "I", "N", "UP", "B", "A", "C4", "T20"]
lint.ignore = ["E501", "W293"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true
disable_error_code = ["import-untyped"]

[project.optional-dependencies]
dev = [
    "pytest",
]
