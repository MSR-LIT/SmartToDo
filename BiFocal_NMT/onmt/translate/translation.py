""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters.text_dataset import TextMultiField


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField)
        self._has_text_qry = isinstance(
            dict(self.fields)["qry"], TextMultiField)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    # The conversion from probability indices to raw tokens happen here !
    # For the indices within target vocab, we just use the argmax. This is why collapse scores was useful.
    # For out of vocab tokens, we decode using qry_vocab and src_vocab.
    def _build_target_tokens(self, qvocab, qry_vocab, src_vocab, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        vocab = tgt_field.vocab
        qry_vocab_size = qvocab

        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            # Modified to accommodate qry and src copy.
            elif tok >= len(vocab) and tok < len(vocab)+qry_vocab_size:
                tokens.append(qry_vocab.itos[tok - len(vocab)])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)-qry_vocab_size])

            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break

        # Exclude this if 'part' and set replace_unk as 'false'. Generation of unk tokens outside qry, src is rare.
        # if self.replace_unk and attn is not None and qry is not None:
        #     for i in range(len(tokens)):
        #         if tokens[i] == tgt_field.unk_token:
        #             _, max_index = attn[i][:len(qry_raw)].max(0)
        #             tokens[i] = qry_raw[max_index.item()]
        #             if self.phrase_table != "":
        #                 with open(self.phrase_table, "r") as f:
        #                     for line in f:
        #                         if line.startswith(qry_raw[max_index.item()]):
        #                             tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        if self._has_text_qry:
            qry = batch.qry[0][:, :, 0].index_select(1, perm)
        else:
            qry = None

        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []
        #Find the maximum query vocab size in the batch. This offset will be used to extract argmax tokens.
        _, _, qvocab = batch.qry_map.size()

        for b in range(batch_size):

            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
            else:
                src_vocab = None
                src_raw = None

            if self._has_text_qry:
                qry_vocab = self.data.qry_vocabs[inds[b]] \
                    if self.data.qry_vocabs else None
                qry_raw = self.data.examples[inds[b]].qry[0]
            else:
                qry_vocab = None
                qry_raw = None

            pred_sents = [self._build_target_tokens(
                qvocab, qry_vocab, src_vocab,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]


            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    qvocab, qry_vocab, src_vocab,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(
                src[:, b] if src is not None else None, src_raw,
                qry[:, b] if qry is not None else None, qry_raw,
                pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.  (This is created in dataset_base.Dataset. It is a list of src_ex_vocab.
        src_raw (List[str]): Raw source words.
        qry (LongTensor): Query word IDs.  (This is created in dataset_base.Dataset. It is a list of qry_ex_vocab.
        qry_raw (List[str]): Raw query words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src", "src_raw", "qry", "qry_raw", "pred_sents", "attns", "pred_scores",
                 "gold_sent", "gold_score"]

    def __init__(self, src, src_raw, qry, qry_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.qry = qry
        self.qry_raw = qry_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: Source : {}, Query : {}\n'.format(sent_number, self.src_raw, self.qry_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
