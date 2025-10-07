from setuptools import setup, find_packages

setup(
    name="training-dashboard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "websockets>=12.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "aiosqlite>=0.19.0",
        "psutil>=5.9.8",
        "python-dateutil>=2.8.2",
        "structlog>=24.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "httpx>=0.26.0",
            "black>=24.1.1",
            "ruff>=0.1.14",
            "mypy>=1.8.0",
        ]
    },
    python_requires=">=3.10",
)

