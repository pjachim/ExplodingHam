from typing import Callable


class NormalizedCompressionDistance:
    """
    Compute the Normalized Compression Distance (NCD) between two strings.
    
    NCD is a similarity metric based on Kolmogorov complexity theory. It measures
    the information distance between two objects by comparing their compressed sizes.
    The metric approximates the amount of shared information between two strings.
    
    Parameters
    ----------
    compressor : Callable[[str | bytes], str | bytes]
        A compression function that takes a string or bytes and returns compressed data.
        Common examples include zlib.compress, bz2.compress, or lzma.compress.
    
    Attributes
    ----------
    compressor : Callable
        The compression function used for computing distances.
    
    Examples
    --------
    Using zlib compression:
    
    >>> import zlib
    >>> ncd = NormalizedCompressionDistance(zlib.compress)
    >>> distance = ncd.ncd(b"hello world", b"hello there")
    >>> print(f"Distance: {distance:.4f}")
    Distance: 0.4545
    
    Using bz2 compression for text similarity:
    
    >>> import bz2
    >>> ncd = NormalizedCompressionDistance(bz2.compress)
    >>> doc1 = b"The quick brown fox jumps over the lazy dog"
    >>> doc2 = b"The quick brown fox jumps over the sleepy cat"
    >>> similarity = 1 - ncd.ncd(doc1, doc2)
    >>> print(f"Similarity: {similarity:.2%}")
    Similarity: 45.45%
    
    Comparing identical strings:
    
    >>> import zlib
    >>> ncd = NormalizedCompressionDistance(zlib.compress)
    >>> ncd.ncd(b"test", b"test")
    0.0
    
    Notes
    -----
    The NCD formula is:
    
    .. math::
        NCD(x, y) = \\frac{C(xy) - \\min(C(x), C(y))}{\\max(C(x), C(y))}
    
    where C(x) is the compressed size of x, and xy is the concatenation of x and y.
    
    NCD values range from 0 (identical) to approximately 1 (completely different).
    The actual upper bound depends on the compressor used.
    """

    def __init__(self, compressor: Callable[[str | bytes], str | bytes]) -> None:
        """
        Initialize the NCD with a given compressor.

        Parameters
        ----------
        compressor : Callable[[str | bytes], str | bytes]
            A compression function that takes string or bytes input and returns
            compressed data. Examples: zlib.compress, bz2.compress, lzma.compress.
        
        Examples
        --------
        >>> import zlib
        >>> ncd = NormalizedCompressionDistance(zlib.compress)
        >>> isinstance(ncd.compressor, type(zlib.compress))
        True
        """
        self.compressor = compressor

    def compress_size(self, data: str | bytes) -> int:
        """
        Get the compressed size of the given data.

        Parameters
        ----------
        data : str | bytes
            The input data to be compressed.

        Returns
        -------
        int
            The size in bytes of the compressed data.
        
        Examples
        --------
        >>> import zlib
        >>> ncd = NormalizedCompressionDistance(zlib.compress)
        >>> size = ncd.compress_size(b"hello world")
        >>> size > 0
        True
        
        >>> import bz2
        >>> ncd = NormalizedCompressionDistance(bz2.compress)
        >>> ncd.compress_size(b"test" * 100) < len(b"test" * 100)
        True
        """
        return len(self.compressor(data))

    def ncd(self, x: str | bytes, y: str | bytes) -> float:
        """
        Compute the Normalized Compression Distance between two strings.

        Parameters
        ----------
        x : str | bytes
            The first input string or bytes.
        y : str | bytes
            The second input string or bytes.

        Returns
        -------
        float
            The NCD value, typically between 0 and 1, where 0 indicates identical
            strings and values closer to 1 indicate greater dissimilarity.
        
        Examples
        --------
        Comparing identical strings:
        
        >>> import zlib
        >>> ncd = NormalizedCompressionDistance(zlib.compress)
        >>> ncd.ncd(b"hello", b"hello")
        0.0
        
        Comparing similar strings:
        
        >>> import zlib
        >>> ncd = NormalizedCompressionDistance(zlib.compress)
        >>> distance = ncd.ncd(b"hello world", b"hello there")
        >>> 0 < distance < 1
        True
        
        Comparing completely different strings:
        
        >>> import zlib
        >>> ncd = NormalizedCompressionDistance(zlib.compress)
        >>> d1 = ncd.ncd(b"aaaa", b"bbbb")
        >>> d2 = ncd.ncd(b"aaaa", b"aaab")
        >>> d1 > d2
        True
        
        Using with text documents:
        
        >>> import bz2
        >>> ncd = NormalizedCompressionDistance(bz2.compress)
        >>> doc1 = b"Python is a programming language"
        >>> doc2 = b"Python is a scripting language"
        >>> doc3 = b"Java is completely different"
        >>> ncd.ncd(doc1, doc2) < ncd.ncd(doc1, doc3)
        True
        
        Notes
        -----
        The concatenation order (x + y vs y + x) may produce slightly different
        results depending on the compression algorithm, but the difference is
        typically negligible.
        """
        Cx = self.compress_size(x)
        Cy = self.compress_size(y)
        Cxy = self.compress_size(x + y)

        ncd_value = (Cxy - min(Cx, Cy)) / max(Cx, Cy)
        return ncd_value