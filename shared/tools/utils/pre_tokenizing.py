import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from collections import Counter,defaultdict
import json
import time
from concurrent.futures import ThreadPoolExecutor
import cProfile

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def process_chunk(args):
    filename, start, end, special_tokens = args
    with open(filename, "rb") as f:
        #essa função determina o cursor da leitura do arquivo
        f.seek(start)
        #De acordo com os argumentos passados, se faz o corte(chunk) do arquivo 
        chunk = f.read(end - start)
    chunk = chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="ignore")
    #usando o padrão pedido pela tarefa
    parts = re.split("|".join(re.escape(tok) for tok in special_tokens), chunk)  
    counts = Counter()
    #usando Iterador para nao estourar a memoria.
    for part in parts:
        for match in PAT.finditer(part):
            #.group() é o metodo que retorna a string do token encontrado
            token = match.group()
            token = tuple([bytes([i]) for i in token.encode('utf-8')])
            counts[token] += 1
    return counts
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def process_chunk2(args):
    filename, start, end, special_tokens = args
    with open(filename, "rb") as f:
        #essa função determina o cursor da leitura do arquivo
        f.seek(start)
        #De acordo com os argumentos passados, se faz o corte(chunk) do arquivo 
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #usando o padrão pedido pela tarefa
    parts = re.split("|".join(re.escape(tok) for tok in special_tokens), chunk)        
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    pares_count = Counter()    
    occurrences = defaultdict(set)
    counts = Counter()
    #usando Iterador para nao estourar a memoria.
    for part in parts:
        for match in PAT.finditer(part):
            #.group() é o metodo que retorna a string do token encontrado
            token = match.group()
            token = tuple([bytes([i]) for i in token.encode('utf-8')])
            counts[token] += 1
            for pair in list(zip(token, token[1:])):
                pares_count[pair] += 1
                #occurrences grava para cada par observado, as palavras aonde o par ocorre
                occurrences[pair].add(token)
    
    
    
         
    return pares_count,occurrences, counts
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def parallel_pre_tokenize(filename, special_tokens, num_processes=mp.cpu_count()):
    # encontra os boundaries
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # prepara os argumentos (start, end) para cada processo
    tasks = [(filename, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]   
    
    total_counts = Counter()

    # paralelização
    with mp.Pool(processes=num_processes) as pool:
        for c in pool.map(process_chunk, tasks):
            total_counts.update(c)
    
    return total_counts
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def parallel_pre_tokenize2(filename, special_tokens, num_processes=4):
    # encontra os boundaries
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # prepara os argumentos (start, end) para cada processo
    tasks = [(filename, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # paralelização
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk2, tasks)
    
    # Contagem
    
    #total_counts = next(iter(results))
    total_counts = (Counter(),defaultdict(set),Counter())
    for r in results:
        total_counts[0].update(r[0])
        for k, v in r[1].items():
            total_counts[1][k].update(v)
        total_counts[2].update(r[2])
    return total_counts
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def tokenizer(vocab: dict[int,bytes], pretokens: list[bytes]):
    result = []
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    for pretoken in pretokens:
        tokens = []
        i=0
        while i<len(pretoken):
            for token in sorted_vocab:
                if pretoken.startswith(token,i):
                    match = token
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                print('token não encontrado')
        result.append(tuple(tokens))
    return result
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def atualiza_palavra(old_word: tuple, old_token: tuple[bytes, bytes], new_token: bytes):
    palavra_atualizada = []
    i=0
    while i < len(old_word):
        # verifica se o par começa na posição i
        if i < len(old_word) - 1 and old_word[i] == old_token[0] and old_word[i+1] == old_token[1]:
            palavra_atualizada.append(new_token)
            i += 2  # pula o par
        else:
            palavra_atualizada.append(old_word[i])
            i += 1    
    return tuple(palavra_atualizada)

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

def BPE_tokenizer_trainer(input_path : str, vocab_size : int, special_tokens : list[str]):
    ###### VOCABULARIO INICIAL

    vocab = { id : bytes([id]) for id in range(256)}
    id=256
    for token in special_tokens:
        vocab[id] = token.encode('utf-8')
        id+=1
    
    

    ####### Load data
    
    filename = input_path
    results = parallel_pre_tokenize(filename, special_tokens, num_processes=8)

    ######## Pré-contagem inicial    

    pares_count = Counter()    
    #cria um dicionário onde o elemento-valor padrão é um conjunto(não permite elementos repetidos)
    occurrences = defaultdict(set)    
    #lista de merges
    merges = []
    
    for word,count in results.items():
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pares_count[pair] += count
            #occurrences grava para cada par observado, as palavras aonde o par ocorre
            occurrences[pair].add(word) 
    
    
 
    ####### Treinamento
    for _ in range(vocab_size-len(vocab)):
        # 1. Old_tokens recebe uma tupla com dois tokens com maiores ocorrencias.
        #old_tokens = max(pares_count, key=pares_count.get)
        old_tokens = max(pares_count.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_token = old_tokens[0] + old_tokens[1]
        # Inserção do novo token
        vocab[id] = new_token
        id = id+1
        # Inserção do merge na lista de monitoramento.
        merges.append(old_tokens)

        #2. atualiza só os pares que mudaram
        for word in list(occurrences[old_tokens]):
            # atualiza a palavra com o merge
            # remove contagens antigas
            for i in range(len(word) - 1):
                pair=(word[i], word[i + 1])
                pares_count[pair] -= results[word]
                occurrences[pair].discard(word)
            
            

            new_word = atualiza_palavra(word, old_tokens, new_token)
            results[new_word] = results.pop(word)
            # adiciona contagens novas
            
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pares_count[new_pair] += results[new_word]
                occurrences[new_pair].add(new_word)
               

    
    #lista de ids e tokens    
    # for id, word in enumerate(vocab):
    #     new_vocab[id]=word
    
    return (vocab,merges)
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def BPE_tokenizer_trainer2(input_path : str, vocab_size : int, special_tokens : list[str]):
    ###### VOCABULARIO INICIAL

    vocab = { id : bytes([id]) for id in range(256)}
    id=256
    for token in special_tokens:
        vocab[id] = token.encode('utf-8')
        id+=1
    
    

    ####### Load data
    a = time.time()
    filename = input_path
    total_counts = parallel_pre_tokenize2(filename, special_tokens, num_processes=16)
    pares_count = total_counts[0]
    occurrences = total_counts[1]
    counts = total_counts[2]
    merges = []
    b = time.time()  
    print(f'Tempo de pretokenização e contagem dos pares: {b-a}')
    ####### Treinamento

    a = time.time()
    for _ in range(vocab_size):
        # 1. Old_tokens recebe uma tupla com dois tokens com maiores ocorrencias.
        old_tokens = max(pares_count, key=pares_count.get)
        new_token = old_tokens[0] + old_tokens[1]
        # Inserção do novo token
        vocab[id] = new_token
        id = id +1
        # Inserção do merge na lista de monitoramento.
        merges.append(old_tokens)

        for word in list(occurrences[old_tokens]):
            # with ThreadPoolExecutor(max_workers=2) as executor:

            #     executor.submit(remove_contagens_antigas, pares_count, occurrences, word, counts)

            #     new_word = atualiza_palavra(word, old_tokens, new_token)
            #     counts[new_word] = counts.pop(word)
                
            #     executor.submit(add_contagens_novas, pares_count, occurrences, new_word, counts)
            #remove_contagens_antigas(pares_count, occurrences, word, counts)

            #new_word = atualiza_palavra(word, old_tokens, new_token)
            #counts[new_word] = counts.pop(word)

            #add_contagens_novas(pares_count, occurrences, new_word, counts)
            for pair in list(zip(word, word[1:])):
                pares_count[pair] -= counts[word]
                occurrences[pair].discard(word)
            
            new_word = atualiza_palavra(word, old_tokens, new_token)
            counts[new_word] = counts.pop(word)

            for pair in list(zip(new_word, new_word[1:])):
                pares_count[pair] += counts[new_word]
                occurrences[pair].add(new_word)
    b = time.time()  
    print(f'Tempo de treinamento: {b-a}') 



    
    #lista de ids e tokens    
    # for id, word in enumerate(vocab):
    #     new_vocab[id]=word
    
    return (vocab,merges)
            
def remove_contagens_antigas(pares_count: Counter, occurrences: defaultdict, word, counts: Counter):    
    for pair in list(zip(word, word[1:])):
        pares_count[pair] -= counts[word]
        occurrences[pair].discard(word)

def add_contagens_novas(pares_count: Counter, occurrences: defaultdict, new_word, counts: Counter):    
    for pair in list(zip(new_word, new_word[1:])):
        pares_count[pair] += counts[new_word]
        occurrences[pair].add(new_word)

    


if __name__ == "__main__":
    # start_time1 = time.time()
    # a = parallel_pre_tokenize('data\TinyStoriesV2-GPT4-valid.txt',['<|endoftext|>'],num_processes=12)
    # end_time1 = time.time()

    # start_time2 = time.time()
    # b = parallel_pre_tokenize2('data\TinyStoriesV2-GPT4-valid.txt',['<|endoftext|>'],num_processes=12)
    # end_time2 = time.time()

    start_time1 = time.time()
    BPE_tokenizer_trainer('data\TinyStoriesV2-GPT4-valid.txt',10000, ['<|endoftext|>'])
    end_time1 = time.time()
    start_time2 = time.time()
    BPE_tokenizer_trainer2("data\TinyStoriesV2-GPT4-valid.txt",10000, ["<|endoftext|>"])
    end_time2 = time.time()
    print(f'end_time1 - start_time1={end_time1-start_time1}')
    print(f'end_time2 - start_time2={end_time2-start_time2}')
    # print(a == b)



