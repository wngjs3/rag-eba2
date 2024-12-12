import os
import logging
from dotenv import load_dotenv

import lib.bedrock as bedrock
import lib.extractpdf as extractpdf
import lib.opensearch as opensearch
from lib.logging_config import setup_logging

# load .env
load_dotenv(override=True)

# Logging
setup_logging()
logger = logging.getLogger(__name__)


def preprocessing():
    # Extract images and metadata
    pdffile = "./pdf/dummy.pdf"
    savedir = "./images_mu"
    extractpdf.extract_images_and_metadata(pdffile, savedir)

    # Create a Bedrock session using default credentials
    bedrock_session = bedrock.get_bedrock_session(
        region_name=os.getenv("AWS_REGION", "ap-northeast-2")
    )

    # Get the model ID for Claude 3.5 Sonnet
    bedrock_modelid = os.getenv("BEDROCK_MODEL_ID")

    # Extract images, captions, and metadata from the PDF using Claude 3.5 Sonnet
    metadata_file = extractpdf.extract_images_caption_and_metadata(
        pdffile, savedir, bedrock_session=bedrock_session, bedrock_modelid=bedrock_modelid)


def insert_to_opensearch():
    savedir = "./images_mu"

    # Create a Bedrock session for the default AWS credentials
    bedrock_session = bedrock.get_bedrock_session(
        region_name=os.getenv("AWS_REGION", "ap-northeast-2")
    )

    # Insert the extracted metadata into OpenSearch
    metadata_file = savedir + "/metadata.json"
    opensearch.insert_metadata_to_opensearch(
        metadata_file, bedrock_session, os.getenv("OPENSEARCH_ENDPOINT"), os.getenv("OPENSEARCH_INDEX_NAME"))

preprocessing()
insert_to_opensearch()
