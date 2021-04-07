# Imports Python
import os
import multiprocessing as mp
import json
import pandas as pd
import shelve
from tqdm.notebook import tqdm
# Imports Gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.doc2vec import Doc2Vec
from gensim.models.coherencemodel import CoherenceModel
from gensim.similarities.docsim import Similarity
# Imports Twins
from twins.utils import Doc2VecCorpus

class Models:
    '''
    Implementação de modelos de análise de documentos para fins de treinamento e geração de vetores para os documentos dos
    corpus. Os vetores gerados a partir destes modelos serão usados para as comparações de similaridade.
    Parâmetros:
        corpus (Corpus ou CorpusDimensao) --> O corpus que será usado nos modelos
    Atributos:
        corpus (Corpus ou CorpusDimensao) --> O corpus que será usado nos modelos
    '''

    def __init__(self, corpus):
        self.corpus = corpus
        self._modelos = {'tfidf': {}, 'tfidf_pivot': {}, 'lsi': {}, 'lda': {}, 'doc2vec': {}}
        self._exts = {'tfidf': 'bow2tfidf', 'tfidf_pivot': 'bow2tfidf_pivot', 'lsi': 'tfidf2lsi', 'lda': 'bow2lda', 'doc2vec': 'doc2vec'}
        self._shelf = f'models_{self.corpus._link_nome}'
        self._arqs = {'modelos': {}, 'indices':{}}
        # Recupera as configurações anteriores
        self._iniciar_models()

    def __contains__(self, modelo):
        return modelo in self._modelos

    def __iter__(self):
        for modelo in self._modelos:
            return self.__getitem__(modelo)

    def __getitem__(self, modelo):
        '''
        Retorna o modelo correspondente.
        Parâmetros:
            modelo (str) --> Indicador do modelo que pode ser "tfidf", "tfidf_pivot", "lsi", "lda" ou "doc2vec"
        Retorno: o modelo solicitado, se existir
        '''
        if not os.path.isfile(self._arqs['modelos'][modelo]):
            print(f'O modelo "{modelo} não foi implementado ou montado."')
            return None
        if modelo in ['tfidf', 'tfidf_pivot']: model = TfidfModel.load(self._arqs['modelos'][modelo])
        elif modelo == 'lsi': model = LsiModel.load(self._arqs['modelos'][modelo])
        elif modelo == 'lda': model = LdaModel.load(self._arqs['modelos'][modelo])
        elif modelo == 'doc2vec': model = Doc2Vec.load(self._arqs['modelos'][modelo])
        return model

    def __len__(self):
        return len(self._modelos)

    def ajustar_modelo(self, modelo, **kwargs):
        '''
        Realiza os ajustes dos hiperparâmetros e pesos do modelo indicado.
        Retorno: None
        '''
        if modelo not in self._modelos:
            print(f'O modelo "{modelo}" não foi implementado.')
            return
        for k, v in kwargs.items():
            if k not in self._modelos[modelo]:
                print(f'O parâmetro "{k}" não faz parte do modelo "{modelo}".')
                continue
            self._modelos[modelo][k] = v
        self._salvar_models()

    def gerar_modelo(self, modelo):
        '''
        Treina o modelo selecionado, salvando-o. Após, cria a matrix de similaridade para o corpus transformado.
        Parâmetros:
            modelo (str) --> nome do modelo: "tfidf", "tfidf_pivot", "lsi", "lda" ou "doc2vec"
        Retorno: None
        '''
        # Verifica se o modelo foi implementado
        if modelo not in self._modelos:
            print(f'O modelo "{modelo}" não foi implementado.')
            return
        # Define os nomes dos arquivos
        arq_model = os.path.join(self.corpus._pastas['modelos'], f'{self.corpus._link_nome}.{self._exts[modelo]}')
        arq_index = os.path.join(self.corpus._pastas['indices'], f'{self.corpus._link_nome}_{modelo}.idx')
        # Gera o modelo solicitado
        if modelo == 'tfidf':
            # Inicializa o modelo
            corpus_train = self.corpus.corpus(tipo='bow')
            num_features = self.corpus.num_tokens
            model = TfidfModel(corpus=corpus_train
                              ,id2word=self.corpus.dicionario())
        elif modelo == 'tfidf_pivot':
            # Inicializa o modelo
            corpus_train = self.corpus.corpus(tipo='bow')
            num_features = self.corpus.num_tokens
            model = TfidfModel(corpus=corpus_train
                              ,id2word=self.corpus.dicionario()
                              ,smartirs='nfu'
                              ,pivot = self.corpus.num_tokens / self.corpus.num_docs)
        elif modelo == 'lda':
            # Inicializa o modelo
            corpus_train = self.corpus.corpus(tipo='bow')
            num_features = self._modelos[modelo]['num_topics']
            model = LdaModel(corpus=corpus_train
                            ,id2word=self.corpus.dicionario()
                            ,num_topics=num_features)
        elif modelo == 'lsi':
            # Inicia o modelo
            corpus_train = self.corpus.corpus(tipo='tfidf')
            num_features = self._modelos[modelo]['num_topics']
            model = LsiModel(corpus=corpus_train
                            ,id2word=self.corpus.dicionario()
                            ,num_topics=num_features)
        elif modelo == 'doc2vec':
            # Instancia o modelo Doc2Vec
            corpus_train = self.corpus.corpus(tipo='tagged')
            num_features = self._modelos[modelo]['vector_size']
            model = Doc2Vec(vector_size=num_features
                           ,workers=mp.cpu_count()/2
                           ,alpha=self._modelos[modelo]['alpha']
                           ,min_alpha=self._modelos[modelo]['min_alpha'])
            # Obtém o vocabulário do corpus para treinar o modelo Doc2Vec
            model.build_vocab(corpus_train)
            # Treina o modelo Doc2Vec
            model.train(corpus_train, total_examples=model.corpus_count, epochs=model.epochs)
        else:
            print(f'O modelo "{modelo}" não foi implementado.')
            return
        # Salva o modelo treinado
        model.save(self._arqs['modelos'][modelo])
        # Define o corpus para a matriz de similaridade
        if modelo == 'doc2vec': corpus = Doc2VecCorpus(model)
        else: corpus = model[corpus_train]
        # Gera o index a partir do modelo serializado
        index = Similarity(output_prefix=self._arqs['indices'][modelo], corpus=corpus, num_features=num_features)
        # Salva o índice
        index.save(self._arqs['indices'][modelo])

    def parametros(self):
        '''
        Retorna o dicionário com os parâmetros dos modelos, sendo a chave o nome do modelo e o valor um dicionário com
        os parâmetros de cada modelo.
        Retorno: dicionário com os parâmetros dos modelos.
        '''
        return self._modelos

    def semelhantes(self, id_ficha_query, teste=False):
        '''
        Pesquisa no corpus quais fichas tem características mais semelhantes às da ficha indicada.
        Parâmetros:
            id_ficha_query (int) --> Identificador da ficha que servirá de comparação para buscar as semelhantes
        Retorno: um Pandas DataFrame na ordem decrescente de semelhança das fichas
        '''
        primeiro = True
        for modelo in self._modelos:
            # Obtém o peso do modelo
            peso = self._modelos[modelo]['peso']
            # Obtém a matriz de similaridade do modelo
            index = Similarity.load(self._arqs['indices'][modelo])
            sims = index.similarity_by_id(id_ficha_query)
            # Cria um dicionário com o resultado da query para o modelo
            sims_dict = {'ficha': [], 'per_sim': [], 'peso': []}
            for id_ficha, per_sim in enumerate(sims):
                if id_ficha == id_ficha_query: continue
                if not teste and per_sim < self._modelos[modelo]['min_per_sim']: continue
                sims_dict['ficha'].append(id_ficha)
                sims_dict['per_sim'].append(round(per_sim * 100, 2))
                sims_dict['peso'].append(peso)
            # Monta o DataFrame com o resultado da query
            # Se não for o primeiro modelo, junta com o resultado anterior
            if primeiro:
                resultado = pd.DataFrame(data=sims_dict)
                resultado.set_index('ficha', inplace=True)
                resultado.columns = pd.MultiIndex.from_product([[modelo], ['per_sim', 'peso']])
                resultado.sort_values(by=(modelo, 'per_sim'), ascending=False, inplace=True)
                resultado[(modelo, 'ordem')] = [i for i in range(1, resultado.shape[0]+1)]
                resultado = resultado.astype({(modelo, 'ordem'): 'int64'})
                primeiro = False
            else:
                parcial = pd.DataFrame(data=sims_dict)
                parcial.set_index('ficha', inplace=True)
                parcial.columns = pd.MultiIndex.from_product([[modelo], ['per_sim', 'peso']])
                parcial.sort_values(by=(modelo, 'per_sim'), ascending=False, inplace=True)
                parcial[(modelo, 'ordem')] = [i for i in range(1, parcial.shape[0]+1)]
                parcial = parcial.astype({(modelo, 'ordem'): 'int64'})
                resultado = resultado.join(parcial, how='outer')
        # Preenches com zeros os valores não encontrados em cada modelo
        resultado.fillna(0, inplace=True)
        # Calcula a probabilidade geral com base no peso de cada modelo e ordena por esse valor em ordem decrescente
        geral = ('geral', 'per_sim')
        resultado[geral] = [0 for i in range(resultado.shape[0])]
        peso_total = 0
        for modelo in self._modelos:
            resultado[geral] = resultado[geral] + (resultado[(modelo, 'per_sim')] * resultado[(modelo, 'peso')])
            peso_total += self._modelos[modelo]['peso']
        resultado[geral] = round(resultado[geral] / peso_total, 2)
        resultado.sort_values(by=geral, ascending=False, inplace=True)
        resultado[('geral', 'ordem')] = [i for i in range(1, resultado.shape[0]+1)]
        return resultado

    def testar_num_topics(self, modelo, num_topicos=[20, 50, 100, 200, 300, 400, 500, 1000, 1500]
                         ,perc_fichas=0.2, vetor_testes=None, tipo_teste='similaridade'):
        '''
        Testa a coerência dos modelos gerados por tópicos para uma lista de quantidade de tópicos para encontrar
        o melhor número de tópicos para o modelo com relação ao corpus.
        Parâmetros:
            modelo (str) --> Modelo a ser testado: "lda", "lsi" ou "doc2vec".
            num_topicos (list de int) --> Lista de números de tópicos a serem testados
                    (default: [20, 50, 100, 200, 300, 400, 500, 1000, 1500])
            per_fichas (float) --> Percentual de fichas do corpus a serem considerados para o teste (default: 0.2)
            vetor_teste (list de tuple) --> Lista de pares de fichas para testes de similaridade. É ignorado se o teste
                é o "u_mass" (default: None)
            tipo_testes (str) --> Tipo de teste: "u_mass" ou "similaridade" (default: "similaridade")
        Retorno: um dicionário de dicionários. A chave do dicionário principal é o número de tópicos e, para cada número de
            tópicos, há outro dicionário com as seguintes chaves:
                "medida" --> Valor de coerência calculado para o modelo com aquele número de tópicos.
                "modelo" --> O modelo gerado para aquele número de tópicos
        '''
        # Verifica se o teste para o modelo foi implantado
        if modelo not in ['lda', 'lsi', 'doc2vec']:
            print(f'O modelo {modelo} ou não é de tópico ou não foi implantado.')
            return
        if tipo_teste not in ['u_mass', 'similaridade']:
            print(f'O tipo de teste {tipo_teste} não foi implementado.')
            return
        if modelo == 'doc2vec' and tipo_teste == 'u_mass':
            print('O teste de coerência com u_mass não pode ser usado para o modelo doc2vec.')
            return
        # Iniciando as variáveis para os testes
        resultado = {}
        arq_index = os.path.join(self.corpus._pastas['indices'], f'{self.corpus._link_nome}_testes.idx')
        if vetor_testes:
            flat = list(zip(*vetor_testes))
            fichas_incluir = set(flat[0])
            fichas_incluir.update(flat[1])
        else: fichas_incluir = None
        # Define os corpus de treinamento e o corpus parcial
        if modelo == 'lsi':
            bow = self.corpus.corpus(tipo='bow')
            corpus_parcial = bow.fatiar(perc_fichas=perc_fichas, incluir=fichas_incluir)
            model_tfidf = self['tfidf'] or TfidfModel(corpus=corpus_parcial, id2word=self.corpus.dicionario())
            corpus_train = model_tfidf[corpus_parcial]
        elif modelo == 'lda':
            bow = self.corpus.corpus(tipo='bow')
            corpus_parcial = corpus_train = bow.fatiar(perc_fichas=perc_fichas, incluir=fichas_incluir)
        elif modelo == 'doc2vec':
            corpus_tagged = self.corpus.corpus(tipo='tagged')
            corpus_parcial = corpus_train = corpus_tagged.fatiar(perc_fichas=perc_fichas, incluir=fichas_incluir)
        # Obtém a relação dos ids_fichas do corpus parcial
        if fichas_incluir: ids_fichas = corpus_parcial.fichas()
        else: ids_fichas = list(range(len(corpus_parcial)))
        # Faz o teste para cada quantidade de tópicos
        for num in tqdm(num_topicos):
            print(f'Criando modelo "{modelo}" para num_topics={num}')
            # Treina os modelo solicitado
            if modelo == 'lda':
                model = LdaModel(corpus=corpus_train
                                ,id2word=self.corpus.dicionario()
                                ,num_topics=num)
            elif modelo == 'lsi':
                model = LsiModel(corpus=corpus_train
                                ,id2word=self.corpus.dicionario()
                                ,num_topics=num)
            elif modelo == 'doc2vec':
                model = Doc2Vec(vector_size=num
                               ,workers=mp.cpu_count()/2
                               ,alpha=self._modelos[modelo]['alpha']
                               ,min_alpha=self._modelos[modelo]['min_alpha'])
                # Obtém o vocabulário do corpus para treinar o modelo Doc2Vec
                model.build_vocab(corpus_train)
                # Treina o modelo Doc2Vec
                model.train(corpus_train, total_examples=model.corpus_count, epochs=model.epochs)
            # Salva o modelo construído para o número de tópicos da iteração
            resultado[num] = {'modelo': model}
            # Realiza o teste de coerência
            if tipo_teste == 'u_mass':
                # Calcula a coerência do modelo para o número de tópicos setado
                print(f'Calculando o score de coerência do modelo "{modelo}" para num_topics={num}')
                cm = CoherenceModel(model=model, corpus=corpus_train, coherence='u_mass')
                resultado[num]['medida'] = cm.get_coherence()
                print(f'Score u_mass = {resultado[num]["medida"]}')
            # Realiza o teste de similaridade
            elif tipo_teste == 'similaridade':
                # Define o corpus para a matriz de similaridade
                if modelo == 'doc2vec': corpus = Doc2VecCorpus(model)
                else: corpus = model[corpus_train]
                # Calcula a similaridade do modelo para o número de tópicos setado
                print(f'Calculando o score de similaridade do modelo "{modelo}" para num_topics={num}')
                index = Similarity(output_prefix=arq_index, corpus=corpus, num_features=num)
                medidas = []
                for ficha_query, ficha_target in vetor_testes:
                    id_query = self.corpus.ficha2id(ficha_query)
                    query = ids_fichas.index(id_query)
                    id_target = self.corpus.ficha2id(ficha_target)
                    target = ids_fichas.index(id_target)
                    posicao, _ = self._obter_posicao_target(index, query, target)
                    medidas.append(1 / posicao)
                valores = pd.Series(medidas)
                resultado[num]['medida'] = valores.median()
                print(f'Score similaridade = {resultado[num]["medida"]}')
        return resultado

    def testar_modelo(self, modelo, vetor_testes, sucesso=100):
        '''
        Realiza testes no modelo, buscando a similaridade da dupla de teste.
        Parâmetros:
            modelo (str) --> Nome do modelo que se deseja pesquisar
            vetor_testes (tuple (str, str)) --> Dupla de fichas, sendo a primeira o argumento de pesquisa e a segunda a ficha
                cuja semelhança se espera encontrar
            sucesso (int) --> Posição máxima na qual pode estar a ficha cuja semelhança se deseja encontrar para ser considerado
                que o modelo obteve sucesso na pesquisa (default: 100)
        Retorno: um DataFrame Pandas com as seguintes colunas:
            * query --> A ficha que foi usada para fazer a pesquisa de semelhança (str)
            * target --> A ficha que foi procurada no resultado da pesquisa (str)
            * ordem --> A posição na qual a ficha target foi encontrada (int)
            * per_sim --> A probabilidade de semelhança apontada pelo modelo (float)
            * sucesso --> Indicador de sucesso ou não da pesquisa (boolean)
        '''
        # Obtém a matriz de similaridade do modelo
        index = Similarity.load(self._arqs['indices'][modelo])
        # Inicia o dicionário que será usado para montar o DataFrame
        resultados = {'query': [], 'target': [], 'ordem': [], 'per_sim': [], 'sucesso': []}
        # Percorre o vetor de testes para realizar as pesquisas
        for query, target in vetor_testes:
            resultados['query'].append(query)
            resultados['target'].append(target)
            id_query = self.corpus.ficha2id(query)
            id_target = self.corpus.ficha2id(target)
            posicao, per_sim = self._obter_posicao_target(index, id_query, id_target)
            resultados['ordem'].append(posicao)
            resultados['per_sim'].append(per_sim)
            resultados['sucesso'].append(posicao <= sucesso)
        return pd.DataFrame(data=resultados)

    def tipos_modelos(self):
        '''
        Retorna uma lista com os modelos implementados.
        Retorno: a lista com os nomes dos modelos implementados (list de str)
        '''
        return list(self._modelos.keys())

    def _iniciar_models(self):
        '''
        Faz as configurações iniciais do objeto e recupera os dados anteriormente salvos
        Retorno: None
        '''
        # Monta o dicionário default de modelos
        for modelo in self._modelos:
            self._modelos[modelo] = {'peso': 1.0, 'min_per_sim': 0.4}
            if modelo in ['lsi', 'lda']:
                self._modelos[modelo]['num_topics'] = 300
            if modelo == 'doc2vec':
                self._modelos[modelo]['vector_size'] = 300
                self._modelos[modelo]['alpha'] = 0.055
                self._modelos[modelo]['min_alpha'] = 0.005
        # Montando o dicionário com os endereços dos arquivos dos modelos
        for modelo, ext in self._exts.items():
            self._arqs['modelos'][modelo] = os.path.join(self.corpus._pastas['modelos'],
                                                         f'{self.corpus._link_nome}.{ext}')
            self._arqs['indices'][modelo] = os.path.join(self.corpus._pastas['indices'],
                                                         f'{self.corpus._link_nome}_{modelo}.idx')
        # Verifica se há um arquivo shelve criado
        if not os.path.isfile(f'{self.corpus._arqs["shelve"]}.dat'): return
        with shelve.open(self.corpus._arqs["shelve"]) as db:
            # Verifica se a chave dos modelos do corpus está no arquivo
            if self._shelf not in db: dados = None
            else: dados = db[self._shelf]
        if dados:
            # Ajusta os parâmetros dos modelos conforme os valores salvos anteriormente
            for modelo in dados:
                self.ajustar_modelo(modelo=modelo, **dados[modelo])
        # Salva uma nova versão das configurações para o caso de mudança de versão da classe
        self._salvar_models()

    def _obter_posicao_target(self, index, id_query, id_target):
        '''
        Pesquisa pelo id_query na matriz de similaridades sims e retorna a posição de id_target considerando a ordem
        decrescente das probabilidades de semelhança.
        Parâmetros:
            index (Similarity) --> Matriz de similaridade de Gensim
            id_query (int) --> Índice do vetor no corpus cujas similaridades se pretende encontrar
            id_target (int) --> Índice do vetor no corpus cuja posição na ordem inversa das probabilidades se deseja encontrar
        Retorno: uma tupla onde o primeiro elemento é a posição de id_target na resposta da consulta de similaridade ordenada
            inversamente pelas probabilidades e o segundo é a probabilidade de similaridade do id_target (tuple (int, float))
        '''
        sims = index.similarity_by_id(id_query)
        sims_df = pd.DataFrame(sims, columns=['per_sim'])
        sims_df.sort_values(by='per_sim', ascending=False, inplace=True)
        sims_df['ordem'] = [i for i in range(1, sims_df.shape[0]+1)]
        if id_target not in sims_df.index: ordem = per_sim = None
        else: ordem, per_sim = sims_df.loc[id_target, 'ordem'], sims_df.loc[id_target, 'per_sim']
        return ordem, per_sim

    def _salvar_models(self):
        '''
        Persiste os dados do modelo.
        Retorno: None
        '''
        with shelve.open(self.corpus._arqs["shelve"]) as db:
            db[self._shelf] = self._modelos
