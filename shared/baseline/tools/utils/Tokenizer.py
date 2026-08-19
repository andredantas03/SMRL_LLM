import regex as re
from collections import Counter,defaultdict
from typing import Iterable, Iterator
import pickle


class Tokenizer():
    def __init__(self,vocab: dict[int,bytes],merges: list[tuple[bytes, bytes]], special_tokens : list[str] | None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.OUT_OF_VOCAB = len(self.merges)+1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens: list[str] | None=None):
        with open(vocab_filepath,'rb') as f:
            vocab =  pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges = pickle.load(f)
        
        return Tokenizer(vocab=vocab, merges = merges, special_tokens=special_tokens)    
    
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int] :
        for text in iterable:            
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids:list[int])-> str:
        entrada = b''.join(self.vocab[x] for x in ids)
        return entrada.decode("utf-8", errors="replace")   

    def pretokenizer(self, text: str) -> Iterator[str]:
        #Se existirem special tokens,partes = split
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens,reverse = True)
            # para manter os delimitadores usa-se '(padrão)' 
            pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            partes = re.split(pattern, text)
        else:
            partes = [text]    
        for p in partes:
            if p!='': 
                if p in self.special_tokens:
                    for tok in self.special_tokens:
                        if p == tok: yield [p.encode('utf-8')]
                else:
                    for match in self.PAT.finditer(p):
                        saida = [bytes([i]) for i in match.group().encode("utf-8")]
                        yield saida
             
    def encode(self,text: Iterator[str])-> list[int]:

        #Desacopla do objeto
        ranks = self.ranks

        #Desacopla do objeto
        get = ranks.get
        
        #Desacopla do objeto
        token_to_id = self.token_to_id
        
        #Desacopla do objeto
        OOV = self.OUT_OF_VOCAB
        
        #Se texto for nulo, retorna lista vazia
        if not text: return []

        #Instancia gerador de pretokens
        pretokens = self.pretokenizer(text)

        #Lista vazia de ids
        ids = []

        for word in pretokens:
            #Se palavra é special token
            if word in self.special_tokens:
                ids.append(token_to_id[word[0]])
                
            #Se palavra menor que 2 caracteres
            if len(word) < 2:
                ids.append(token_to_id[word[0]])

            #Todos os outros casos    
            else:
                #Loop infinito ate nao haver mais pares mapeados em self.merges
                while True:
                    #Cria todos os pares possiveis com os tokens da palavra
                    pairs_ranked = {(i, i+1): get((word[i], word[i+1]),OOV) for i in range(len(word)-1)}

                    #Se o tamanho de pairs_ranked é zero indica que não há mais trocas a serem feitas
                    if len(pairs_ranked)==0: break   

                    #Elege o par criado primeiro
                    primeiro_par = min(pairs_ranked, key=pairs_ranked.get)

                    #Se o elemento de pairs_ranked for maior que o tamanho do vocabulario, indica que não ha mais merges a se fazer
                    if pairs_ranked[primeiro_par]>=OOV: break

                    #atribui-se o ponteiro para i
                    i = primeiro_par[0]

                    #Realiza o merge dos tokens
                    word[i] = word[i]+word[i+1]

                    #Libera da memoria o ultimo token, ja que acabou de ser mergeado
                    del word[i+1]
                
                #Ao final do loop, reune-se todos os ids para saida
                for token in word:
                    ids.append(token_to_id[token])

        
        return ids    
   