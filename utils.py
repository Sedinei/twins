# Imports Python
import csv
import sqlite3
import os
import re
# Imports Gensim
from gensim.models.doc2vec import TaggedDocument
from gensim import utils as g_utils

RE_ESPACO = re.compile(r'\s+')

def obter_link_name(nome):
    '''
    Deixa o nome passado em minúsculas, sem acentos e com '_' entre as palavras
    Parâmetros:
        nome (String) --> Nome a ser transformado
    Retorno: O nome em minúsculas, sem acentos e com '_' entre as palavras (String)
    '''
    return g_utils.deaccent(RE_ESPACO.sub('_', nome.lower()))

class ConexaoDB:
    '''
    Abstrai a conexão a um banco de dados SQlite3 que é usado para armazenar as informações do corpus e das configurações
    dos objetos. Essa classe é para ser usada em uma estrutura com with, lançando um cursor para a conexão ao DB do arquivo
    e fazendo o commit e close ao final do bloco.
    Parâmetros:
        arq_db (String) --> Endereço onde se encontra o arquivo do DB
    '''
    def __init__(self, arq_db):
        self.arq_db = arq_db
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.arq_db)
        return self.conn.cursor()

    def __exit__(self, tipo_excecao, valor_excecao, traceback):
        self.conn.commit()
        self.conn.close()
        self.conn = None

    def existe(self):
        '''
        Verifica se o arquivo de DB já foi criado.
        Retorno:
            True --> Se existe o arquivo de DB
            False --> Se não existe o arquivo de DB
        '''
        return os.path.isfile(self.arq_db)

class StreamCSV:
    '''
    Essa classe recebe o endereço onde se encontra um dataset armazenado no formato CSV e o transforma em um Stream para a
    formação do corpus. A primeira coluna deve trazer a tag dos documentos que deve ser um número inteiro, enquanto que as
    demais colunas são os atributos que serão tokenizados no corpus.
    A cada iteração, repassa um objeto zip que contém tuplas onde o primeiro elemento da tupla é o nome do atributo e o
    segundo é o valor desse atributo.
    Parâmetros:
        arq_csv (String) --> Endereço onde se encontra o dataset em formato de CSV
        nrows (Int) --> Número máximo de linhas a serem lidas do arquivo. Se None, lê todo o arquivo (default: None)
        start (Int) --> Linha a partir da qual deve iniciar a leitura dos dados do arquivo, iniciando em 0 (default: 0)
        sep (String) --> Separador usado no arquivo CSV (default: ',')
    Atributos:
        reader (csv.reader) --> Objeto de leitura do módulo csv de Python
        nrows (Int) --> Número máximo de linhas a serem lidas do arquivo. Se None, lê todo o arquivo
        qtd_rows (Int) --> Quantidade de linhas de dados do arquivo já lidas 
        atributos (Lista de String) --> Conjunto dos atributos que constam do dataset
    '''
    def __init__(self, arq_csv, nrows=None, start=0, sep=',', encoding='utf-8'):
        # Seta o tamanho máximo do campo para 1GB
        csv.field_size_limit(1073741824)
        self.reader = csv.reader(open(arq_csv, newline='', encoding=encoding)
                                     ,lineterminator='\n', delimiter=sep)
        self.nrows = nrows
        self.qtd_rows = start
        self.atributos = []
        # Obtém o conteúdo de atributos e coloca o reader na linha definida em start
        self._iniciar_objeto()
        
    def _iniciar_objeto(self):
        self.atributos = next(self.reader)
        for i in range(self.qtd_rows): next(self.reader)
    
    def __iter__(self):
        for chunk in self.reader:
            self.qtd_rows += 1
            if self.nrows and self.qtd_rows > self.nrows: return
            yield list(zip(self.atributos, chunk))

class TaggedCorpus:
    '''
    Representa um iterável de TaggedDocument do corpus.
    Parâmetros:
        corpus (Corpus ou CorpusDimensao) --> O corpus cujas fichas serão iteradas no formato TaggedDocument
    '''

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for id_ficha in range(self.corpus.num_fichas):
            tokens = self.corpus._dao.obter_tokens_id_ficha(id_ficha)
            words = [token for token, freq in tokens.items() for i in range(freq)]
            yield TaggedDocument(words, [id_ficha])

    def __len__(self):
        return self.corpus.num_fichas
    
    def __getitem__(self, id_ficha):
        if isinstance(id_ficha, int):
            tokens = self.corpus._dao.obter_tokens_id_ficha(id_ficha)
            words = [token for token, freq in tokens.items() for i in range(freq)]
            return TaggedDocument(words, [id_ficha])
        if isinstance(id_ficha, slice):
            print('Para slice, use o método "fatiar".')

    def fatiar(self, perc_fichas=0.2, incluir=None):
        '''
        Cria um iterador com um percentual do corpus no formato Tagged
        Parâmetro:
            perc_fichas (float) --> Percentual das fichas do corpus (primeiras n% fichas) (default: 0.2)
            incluir (list de string) --> Lista de ids de fichas a serem incluídas na fatia (default: None)
        '''
        tamanho = int(self.corpus.num_fichas * perc_fichas)
        ids_fichas = list(range(tamanho))
        if incluir:
            for ficha in incluir:
                id_ficha = self.corpus.ficha2id(ficha)
                if id_ficha in ids_fichas: continue
                ids_fichas.append(id_ficha)
        return Fatia(ids_fichas, self.__getitem__)

class Doc2VecCorpus:
    '''
    Iterador que transforma os vetores de documentos de um modelo Doc2Vec em uma lista de listas
    de tuplas (int, float), onde o primeiro elemento é a dimensão do vetor e o segundo é o valor
    do vetor nessa dimensão.
    O iterador lança um vetor por vez (uma lista de tuplas) para alimentar um arquivo no formato
    Market Matrix.
    Parâmetros:
        model (Doc2Vec) --> um modelo Doc2Vec treinado com o corpus que se deseja obter
    Atributos:
        model (Doc2Vec) --> um modelo Doc2Vec treinado com o corpus que se deseja obter
    '''
    def __init__(self, model):
        self.model = model
    
    def __iter__(self):
        for id_ficha in range(self.model.corpus_count):
            yield [(i, v) for i, v in enumerate(self.model.docvecs[id_ficha])]

    def __getitem__(self, id_ficha):
        return [(i, v) for i, v in enumerate(self.model.docvecs[id_ficha])]

class BOWCorpus:
    '''
    Iterável de um corpus no formato BOW.
    Parâmetro:
        corpus (Corpus ou CorpusDimensao) --> O corpus cujas fichas serão iteradas no formato BOW
    '''
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for id_ficha in range(self.corpus.num_fichas):
            yield self.corpus._dao.obter_bow_id_ficha(id_ficha)

    def __len__(self):
        return self.corpus.num_fichas

    def __getitem__(self, id_ficha):
        if isinstance(id_ficha, int):
            return self.corpus._dao.obter_bow_id_ficha(id_ficha)
        if isinstance(id_ficha, slice):
            print('Para slice, use o método "fatiar".')

    def fatiar(self, perc_fichas=0.2, incluir=None):
        '''
        Cria um iterador com um percentual do corpus no formato BOW
        Parâmetro:
            perc_fichas (float) --> Percentual das fichas do corpus (primeiras n% fichas) (default: 0.2)
            incluir (list de string) --> Lista de fichas a serem incluídas na fatia (default: None)
        '''
        tamanho = int(self.corpus.num_fichas * perc_fichas)
        ids_fichas = list(range(tamanho))
        if incluir:
            for ficha in incluir:
                id_ficha = self.corpus.ficha2id(ficha)
                if id_ficha in ids_fichas: continue
                ids_fichas.append(id_ficha)
        return Fatia(ids_fichas, self.__getitem__)

class Fatia:
    '''
    Classe auxiliar para implementar um iterador de parte de um corpus
    Parâmetros:
        ids_fichas (list de int) --> Lista dos ids das fichas a serem retornadas
        funcao (FunctionObject) --> Uma função que retorna o valor do objeto para um detarminado índice
    Atributos:
        tamanho (int) --> Quantidade de fichas a serem retornadas
        funcao (FunctionObject) --> Uma função que retorna o valor do objeto para um detarminado índice
    '''
    def __init__(self, ids_fichas, funcao):
        self.ids_fichas = ids_fichas
        self.funcao = funcao

    def __getitem__(self, id_fatia):
        id_ficha = self.ids_fichas[id_fatia]
        return self.funcao(id_ficha)

    def __len__(self):
        return len(self.ids_fichas)
        
    def __iter__(self):
        for id_ficha in self.ids_fichas:
            yield self.funcao(id_ficha)

    def fatia2id_ficha(self, id_fatia):
        '''
        A partir do id da fatia (posição do id_ficha na lista interna), retorna o id da ficha no corpus que originou a fatia.
        Parâmetros:
            id_fatia (int) --> Posição da ficha na fatia
        Retorno: Id da ficha no corpus original (int)
        '''
        return self.fichas[id_fatia]
    
    def fichas(self):
        '''
        Retorna a lista dos ids das fichas na ordem que são lançadas.
        Retorno: lista de ids das fichas (list de int)
        '''
        return self.ids_fichas

class FormataDeltatime:
    '''
    Recebe um tempo no formato deltatime e o divide em horas, minutos e segundos.
    Atributos:
        tempo (deltatime) --> O tempo passado para a classe
        horas (int) --> Quantidade de horas medidas
        minutos (int) --> Quantidade de minutos medidos após a última hora
        segundos (int) --> Quantidade de segundos medidos após o último segundo
    '''
    def __init__(self):
        self.tempo = None
        self.horas = 0
        self.minutos = 0
        self.segundos = 0
    
    def __str__(self):
        return f'{self.horas:02}:{self.minutos:02}:{self.segundos:02}'
            
    def formatar(self, tempo):
        self.tempo = tempo
        dias, segundos = self.tempo.days, self.tempo.seconds
        self.horas = dias * 24 + segundos // 3600
        self.minutos = (segundos % 3600) // 60
        self.segundos = (segundos % 60)
