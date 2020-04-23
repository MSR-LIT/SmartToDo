"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
