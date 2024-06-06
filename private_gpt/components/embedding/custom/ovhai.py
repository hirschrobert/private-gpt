# source: https://gist.github.com/gmasse/e1f99339e161f4830df6be5d0095349a
import requests
import time
import logging

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

"""
NB: Make sure you are using a valid token. In the contrary, document indexing will be long due to rate-limite
"""


class OVHcloudAIEEmbeddings(BaseEmbedding):
    _api_key: str = PrivateAttr()
    _api_base: str = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://multilingual-e5-base.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ovhcloud ai endpoints embedding"


    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings from OVHCLOUD AIE.
        Args:
            text: str. An input text sentence or document.
        Returns:
            embeddings: a list of float numbers. Embeddings correspond to your given text.
        """
        headers = {
            "content-type": "text/plain",
            "Authorization": f"Bearer {self._api_key}",
        }

        session = requests.session()
        while True:
            response = session.post(
                self._api_base,
                headers=headers,
                data=text,
            )
            if response.status_code != 200:
                if response.status_code == 429:
                    """Rate limit exceeded, wait for reset"""
                    reset_time = int(response.headers.get("RateLimit-Reset", 0))
                    logging.info("Rate limit exceeded. Waiting %d seconds.", reset_time)
                    if reset_time > 0:
                        time.sleep(reset_time)
                        continue
                    else:
                        """Rate limit reset time has passed, retry immediately"""
                        continue

                """ Handle other non-200 status codes """
                raise ValueError(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            return response.json()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._generate_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._generate_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return [self._generate_embedding(text) for text in texts]