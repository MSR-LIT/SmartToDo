""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      matcher (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, matcher, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.matcher = matcher
        self.decoder = decoder

    def forward(self, src, qry, tgt, src_lengths, qry_lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            qry (Tensor): A query sequence passed to matcher.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            src_lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            qry_lengths(LongTensor): The qry lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, src_memory_bank, src_lengths = self.encoder(src, src_lengths)
        match_state, qry_memory_bank, qry_lengths = self.matcher(qry, qry_lengths)

        if bptt is False:
            self.decoder.init_state(enc_state)

        # attns is a dictionary with keys "std_src", "copy_qry" of different dimensions.
        dec_out, attns = self.decoder(tgt, src_memory_bank, qry_memory_bank,
                                      src_memory_lengths=src_lengths, qry_memory_lengths=qry_lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.matcher.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
