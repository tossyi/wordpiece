# -*- coding: utf-8 -*-
#!/usr/local/bin/python3
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import MeCab


file = "corpus.txt"
outputfile = "corpus-parse.txt"

with open(file,mode='r',encoding='utf-8') as f:
    text = [s.strip() for s in f.readlines()]
f.close()

tagger = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

with open(outputfile,mode='w',encoding='utf-8') as fw:
    for line in text:
        # Morphological analysis
        parseline = tagger.parse(line)
        fw.write(parseline)
fw.close()


bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=32000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

bert_tokenizer.train([outputfile], trainer)

bert_tokenizer.save("./wordpiece.json")