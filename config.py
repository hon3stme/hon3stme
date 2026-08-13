import os

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./football_ai.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "DEMO_KEY")