from functools import lru_cache

import tiktoken


class TiktokenCache:
    """
    Singleton to cache Tiktoken encoding.

    Notes:
    - Tiktoken is chosen for its speed and efficiency.​
    - Tiktoken incorporates commonly used words directly into its vocabulary
      as single tokens, aligning with the concept of word-level tokenization.​
    - Tiktoken supports multiple languages; however, SentencePiece may perform
      better in certain multilingual contexts, we can revisit this in the future.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.encoding = tiktoken.get_encoding("cl100k_base")
        return cls._instance

    @lru_cache
    def encode(self, text: str):
        """
        Cached encoding method with LRU cache.

        Args:
            text (str): Text to encode
        Returns:
            List of tokens
        """
        return self.encoding.encode(text)

    @lru_cache
    def decode_tokens(self, tokens):
        """
        Cached decoding method with LRU cache.

        Args:
            tokens: Tokens to decode
        Returns:
            Decoded text
        """
        return self.encoding.decode_tokens_bytes(tokens)
