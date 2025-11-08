
import boto3
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_aws import ChatBedrock
from settings import Settings
import os

settings = Settings()

# def setup_llm(model_name):
#     """Initialize the language model based on model name."""
#     client=boto3.client(
#                 service_name='bedrock-runtime',
#                 region_name='us-east-1',
#                 aws_access_key_id = settings.AWS_ACCESS_KEY,
#                 aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
#             )
#     if model_name == 'gpt4o':
#         llm = AzureChatOpenAI(
#             azure_endpoint= settings.AZURE_MODEL_ENDPOINT,
#             api_key=settings.AZURE_MODEL_API_KEY,
#             api_version=settings.AZURE_MODEL_API_VERSION,
#             azure_deployment=settings.GPT4O_MODEL,
#             temperature=0.2
#         )
#     elif model_name == 'gpt5':
#         llm = AzureChatOpenAI(
#             azure_endpoint=settings.AZURE_MODEL_ENDPOINT2,
#             api_key=settings.AZURE_MODEL_API_KEY2,
#             api_version=settings.AZURE_MODEL_API_VERSION2,
#             azure_deployment=settings.GPT5_MODEL,
#             # temperature=0.2 gpt5 does not support temperature
#         )   

#     elif model_name == 'claude_3_5_sonnet':
#         llm = ChatBedrock(
#             model_id=settings.CLAUDE_3_5_SONNET_V1_MODEL,
#             client=client,
#             model_kwargs=dict(temperature=0.2)
#         )
#     elif model_name == 'llama_3_1_401B_instruct':
#         llm = ChatOpenAI(
#             model_name=settings.LLAMA_3_1_401B_MODEL,
#             openai_api_key=settings.DATABRICKS_TOKEN,  # Pass API key directly
#             openai_api_base=settings.DATABRICKS_ENDPOINT_URL,  # Use api_base instead of base_url
#             temperature=0.2
#         )
#     elif model_name == 'llama_4_maverick':
#         # Create the ChatOpenAI instance with direct API key and base URL
#         llm = ChatOpenAI(
#             model_name=settings.LLAMA_4_MAVERICK_MODEL,
#             openai_api_key=settings.DATABRICKS_TOKEN,  # Pass API key directly
#             openai_api_base=settings.DATABRICKS_ENDPOINT_URL,  # Use api_base instead of base_url
#             temperature=0.2
#         )
#     elif model_name == 'claude_3_haiku':
#         llm = ChatBedrock(
#             model_id=settings.CLAUDE_3_HAIKU_MODEL,
#             client = client,
#             model_kwargs=dict(temperature=0.2)
#         )
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
    
#     return llm


settings = Settings()

def setup_llm(model_name):
    """Initialize the language model based on model name."""
    
    if model_name == 'gpt4o':
        llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_MODEL_ENDPOINT,
            api_key=settings.AZURE_MODEL_API_KEY,
            api_version=settings.AZURE_MODEL_API_VERSION,
            azure_deployment=settings.GPT4O_MODEL,
            temperature=0.2
        )
    elif model_name == 'gpt5':
        llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_MODEL_ENDPOINT2,
            api_key=settings.AZURE_MODEL_API_KEY2,
            api_version=settings.AZURE_MODEL_API_VERSION2,
            azure_deployment=settings.GPT5_MODEL
        )   

    elif model_name == 'claude_3_5_sonnet':
        # Debug print to verify credentials are loaded
        print(f"AWS Access Key loaded: {bool(settings.AWS_ACCESS_KEY)}")
        print(f"AWS Secret Key loaded: {bool(settings.AWS_SECRET_ACCESS_KEY)}")
        
        # Set environment variables BEFORE creating ChatBedrock
        os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY
        os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        
        # Now create ChatBedrock - it will use the environment variables
        llm = ChatBedrock(
            model_id=settings.CLAUDE_3_5_SONNET_V1_MODEL,
            region_name='us-east-1',
            model_kwargs=dict(temperature=0.2)
        )
        
    elif model_name == 'claude_3_haiku':
        # Set environment variables BEFORE creating ChatBedrock
        os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY
        os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
        os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        
        llm = ChatBedrock(
            model_id=settings.CLAUDE_3_HAIKU_MODEL,
            region_name='us-east-1',
            model_kwargs=dict(temperature=0.2)
        )
        
    elif model_name == 'llama_3_1_401B_instruct':
        llm = ChatOpenAI(
            model_name=settings.LLAMA_3_1_401B_MODEL,
            openai_api_key=settings.DATABRICKS_TOKEN,
            openai_api_base=settings.DATABRICKS_ENDPOINT_URL,
            temperature=0.2
        )
    elif model_name == 'llama_4_maverick':
        llm = ChatOpenAI(
            model_name=settings.LLAMA_4_MAVERICK_MODEL,
            openai_api_key=settings.DATABRICKS_TOKEN,
            openai_api_base=settings.DATABRICKS_ENDPOINT_URL,
            temperature=0.2
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return llm