import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, qry_vocabs=None, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset_qry = len(tgt_vocab)
    _, _, qvocab = batch.qry_map.size()
    offset_src = len(tgt_vocab) + qvocab
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if qry_vocabs is None:
            qry_vocab = batch.qry_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            qry_vocab = qry_vocabs[index]

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        # This portion is changed for using both src and qry copy mechanism.
        for i in range(1, len(qry_vocab)):
            qw = qry_vocab.itos[i]
            ti = tgt_vocab.stoi[qw]
            if ti != 0:
                blank.append(offset_qry + i)
                fill.append(ti)

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset_src + i)
                fill.append(ti)

        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)

        # Optional : Scale ex_vocab scores appropriately, after transfer of within vocab scores. Not implemented.

    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probability of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation  (dec_rnn_size)
       output_size (int): size of output vocabulary    (vocab size)
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn_qry, qry_map, attn_src, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
            ; attn has shape (batch x tlen, src_len)  (resp. batch x tlen, qry_len)
           qry_map (FloatTensor):
               A sparse indicator matrix mapping each query word to
               its index in the "extended" vocab containing.
               ``(qry_len, batch, extra_qry_words)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_src_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn_src.size()
        batch_by_tlen_, qlen = attn_qry.size()
        slen_, batch_size, svocab = src_map.size()
        qlen_, batch_size, qvocab = qry_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)
        aeq(qlen, qlen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn_qry = torch.mul(attn_qry, p_copy)
        mul_attn_src = torch.mul(attn_src, p_copy)

        # qry_map acts a permutation map to allocate the attention scores over correct vocab indices.
        copy_prob_qry = torch.bmm(
            mul_attn_qry.view(-1, batch_size, qlen).transpose(0, 1),
            qry_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob_qry = copy_prob_qry.contiguous().view(-1, qvocab)

        copy_prob_src = torch.bmm(
            mul_attn_src.view(-1, batch_size, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob_src = copy_prob_src.contiguous().view(-1, svocab)

        return torch.cat([out_prob, copy_prob_qry, copy_prob_src], 1)
        # Returned results is (Batch x tgt_len, ext_vocab_size)
        # [ext_vocab_size = V + batch_src_len + 2 + batch_qry_len + 2 (i.e., src_unk, src_pad, qry_unk, qry_pad)]


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, qry_map, align_qry, align_src, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size  : (BT x dynamic_vocab) - 2-dimensional Tensor
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align_qry (LongTensor): ``(batch_size x tgt_len)`` (BT) - 1-dimensional Tensor
            target (LongTensor): ``(batch_size x tgt_len)``    (BT) - 1-dimensional Tensor
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        _, _, qvocab = qry_map.size()

        # probability of tokens copied from source
        copy_ix_qry = align_qry.unsqueeze(1) + self.vocab_size
        copy_ix_src = align_src.unsqueeze(1) + self.vocab_size + qvocab

        copy_tok_probs_qry = scores.gather(1, copy_ix_qry).squeeze(1)
        copy_tok_probs_src = scores.gather(1, copy_ix_src).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs_qry[align_qry == self.unk_index] = 0
        copy_tok_probs_qry += self.eps  # to avoid -inf logs

        copy_tok_probs_src[align_src == self.unk_index] = 0
        copy_tok_probs_src += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = (align_qry == self.unk_index) & (align_src == self.unk_index)

        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs_qry + copy_tok_probs_src + vocab_probs, copy_tok_probs_qry + copy_tok_probs_src
        )

        # This framework would extend when copying from either source or query. But divide by 2 to keep scales same.

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment_qry", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn_qry": attns.get("copy_qry"),
            "align_qry": batch.alignment_qry[range_[0] + 1: range_[1]],
            "copy_attn_src": attns.get("copy_src"),
            "align_src": batch.alignment_src[range_[0] + 1: range_[1]],
        })
        return shard_state

    def _compute_loss(self, batch, output, target, copy_attn_qry, align_qry, copy_attn_src, align_src,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: output of decoder model `[tgt_len x batch x hidden]`
            target: the validate target to compare output with.
            copy_attn_qry: the copy attention value from query.
            align_qry: the align info with qry.
            copy_attn_src: the copy attention value from source.
            align_src: the align info with src.
        """

        target = target.view(-1)
        align_qry = align_qry.view(-1)
        align_src = align_src.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn_qry), batch.qry_map, self._bottle(copy_attn_src), batch.src_map
        )                                                   # Forward of CopyGenerator

        loss = self.criterion(scores, batch.qry_map, align_qry, align_src, target)    # Forward of CopyGeneratorLoss

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        _, _, qvocab = batch.qry_map.size()

        # This portion needs to have offset changed if using both src and qry copy mechanism.
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask_qry = (target_data == unk) & (align_qry != unk)
        correct_mask_src = (target_data == unk) & (align_src != unk)
        offset_align_qry = align_qry[correct_mask_qry] + len(self.tgt_vocab)
        offset_align_src = align_src[correct_mask_src] + len(self.tgt_vocab) + qvocab
        target_data[correct_mask_qry] += offset_align_qry
        target_data[correct_mask_src] += offset_align_src

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
