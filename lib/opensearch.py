import base64
import json
import requests
from requests.auth import HTTPBasicAuth
import logging
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
import time

import lib.bedrock as bedrock
import lib.logging_config as logging_config

logger = logging.getLogger(__name__)


def insert_metadata_to_opensearch(metadata_file, bedrock_session,
                                  opensearch_endpoint, index_name,
                                  region='ap-northeast-2'):
    """
    OpenSearch Serverless에 메타데이터를 삽입하는 함수
    """
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadatas = json.load(f)

    # AWS 인증 설정
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )

    # OpenSearch 클라이언트 생성
    client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint.replace("https://", ""),
            'port': 443
        }],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )

    # 데이터 액세스 규칙이 적용될 때까지 대기
    time.sleep(45)

    documents = []
    for file_name, item in metadatas.items():

        # Extract page number
        item_page_number = item['page']

        # Extract image path
        item_image_file_name = file_name

        # Extract image text
        item_text = item['image_text']

        # Extract image type
        item_type = item['type']

        logger.info(f"item_page_number: {item_page_number}")
        logger.info(f"item_image_file_name: {item_image_file_name}")
        logger.info(f"item_text: {item_text}")
        logger.info(f"item_type: {item_type}")

        # 이미지 데이터를 base64로 인코딩
        with open(item_image_file_name, "rb") as image_file:
            image_data = image_file.read()

        embedding = bedrock.get_text_vector(bedrock_session, item_text)

        # 문서 생성
        document = {
            "page_number": int(item_page_number),
            "image_file_name": item_image_file_name,
            "text": item_text,
            "image_type": item_type,
            "image": base64.b64encode(image_data).decode('utf-8'),
        }
        if embedding is not None:
            document["content_vector"] = embedding

        # logger.info(f"document: {document}")

        # 문서 인덱싱
        response = client.index(
            index=index_name,
            body=document
        )

        # 결과 출력
        logger.info(f"Document indexing status: {response}")


def query_imagesearch_to_opensearch(query, query_type, doc_count=5, bedrock_session=None,
                                    opensearch_endpoint=None, index_name=None,
                                    region='ap-northeast-2'):
    logger.info(f"Starting query_imagesearch_to_opensearch with query: {query}, doc_count: {doc_count}")

    if opensearch_endpoint is None or index_name is None:
        logger.error("opensearch_endpoint and index_name must be provided")
        return [], []

    # AWS 인증 설정
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )

    # OpenSearch 클라이언트 생성
    client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint.replace("https://", ""),
            'port': 443
        }],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )

    # Query body 생성
    vector_query = bedrock.get_text_vector(bedrock_session, query)
    logger.info(f"Vector query generated: {len(vector_query)} dimensions")
    
    if (query_type == "imagesearch"):
        query_body = {
            "size": doc_count,
            "_source": {"excludes": ["content_vector"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "image_type": "sub"
                            }
                        },
                        {
                            "knn": {
                                "content_vector": {
                                    "vector": vector_query,
                                    "k": 5
                                }
                            }
                        }
                    ]
                }
            }
        }
    else:
        query_body = {
            "size": doc_count,
            "_source": {"excludes": ["content_vector"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "image_type": "main"
                            }
                        },
                        {
                            "knn": {
                                "content_vector": {
                                    "vector": vector_query,
                                    "k": 5
                                }
                            }
                        }
                    ]
                }
            }
        }
    # logger.info(f"Query body: {json.dumps(query_body, indent=2)}")

    # OpenSearch 검색 실행
    response = client.search(
        index=index_name,
        body=query_body
    )
    logger.info(f"Response status: {response}")

    # Process response
    images = []
    contents = []
    if 'hits' in response and 'hits' in response['hits']:
        for hit in response['hits']['hits']:
            image_binary = hit['_source']['image']
            images.append(image_binary)
            content = hit['_source']['text']
            contents.append(content)

        logger.info(f"Number of images retrieved: {len(images)}")
        logger.info(f"Number of contents retrieved: {len(contents)}")
        return images, contents
    else:
        logger.error(f"Error in OpenSearch query")
        logger.error(f"Error response: {response}")
        return [], []
