from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AZURE_MODEL_API_KEY: str
    AZURE_MODEL_API_VERSION: str
    AZURE_MODEL_ENDPOINT: str

    AZURE_MODEL_API_KEY2: str
    AZURE_MODEL_API_VERSION2: str
    AZURE_MODEL_ENDPOINT2: str

    AWS_ACCESS_KEY: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    # AWS_SESSION_TOKEN: str

    DATABRICKS_ENDPOINT_URL: str
    DATABRICKS_TOKEN: str

    GPT4O_MODEL: str
    GPT5_MODEL: str
    CLAUDE_3_5_SONNET_V1_MODEL: str
    LLAMA_4_MAVERICK_MODEL: str
    LLAMA_3_1_401B_MODEL: str
    CLAUDE_3_HAIKU_MODEL: str

    VISEVAL_DATASET_PATH: str
    VISEVAL_JSON_PATH: str

    SVG_FILES_PATH: str
    LOG_FOLDER: str

    CHROME_DRIVER_PATH: str
    model_config = SettingsConfigDict(env_file=".env")

