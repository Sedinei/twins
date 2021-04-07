# Imports Python
import pandas as pd
import shelve
import os
from tqdm.notebook import tqdm
# Imports Twins
from twins.dimensoes import Dimensoes
from twins.corpus import Corpus
from twins.utils import obter_link_name

class Twins:
    '''
    Gerenciador do sistema que controla os corpus e é responsável pelas pesquisas e apresentações
    dos resultados.
    Parâmteros:
        projeto (string) --> Nome do projeto no qual estarão os dados, corpus e modelos. Não é case sensitive e ignora acentos.
        corpus_unico (boolean) --> Indica se o corpus será dividido em dimensões (False) ou não (True) (default: True)
    Atributos:
        projeto (string) --> Nome do projeto no qual estarão os dados, corpus e modelos. Não é case sensitive e ignora acentos.
        corpus (Corpus ou Dimensoes) --> O corpus a ser gerenciado. Se corpus_unico for True, recebe um objeto Dimensoes que
                                         é um iterável de objetos CorpusDimensoes. Se corpus_unico for False, recebe um objeto
                                         Corpus.
        corpus_unico (boolean) --> Indica se o corpus será dividido em dimensões (False) ou não (True)
        resultados (dict str:DataFrame) --> Resultados de uma consulta por fichas semelhantes
        max_resultados (int) --> Número máximo de resultados finais a serem apresentados para uma consulta por fichas
                                 semelhantes. Se for igual a 0, não limita o resultado (default: 0)
    '''
    def __init__(self, projeto, corpus_unico=True):
        self.projeto = projeto
        self.corpus = None
        self.corpus_unico = corpus_unico
        self.resultados = {}
        self.max_resultados = 0
        self._arq_shelve = None
        # Inicia o objeto
        self._iniciar_twins()

    def ajustar_pesos_dimensoes(self, **kwargs):
        '''
        Ajusta os pesos das dimensões conforme passado no dicionário.
        Parâmetros:
            O nome da dimensão e o peso que se quer atribuir a ela
        Retorno: None
        '''
        if self.corpus_unico:
            print('Você está trabalhando com um corpus único.')
            return
        self.corpus.ajustar_pesos(**kwargs)

    def dimensoes(self):
        '''
        Lista a relação de dimensões informadas no controle.
        Retorno: lista dos nomes das dimensões (list de string) ou None, se corpus_unico for True.
        '''
        if self.corpus_unico:
            print('Você está trabalhando com um corpus único.')
            return None
        return [dimensao.nome for dimensao in self.corpus]

    def incluir_dimensao(self, dimensao):
        '''
        Faz a inclusão de uma dimensão se corpus_unico for False.
        Parâmetros:
            dimensao (string) --> Nome da dimensão a ser incluída no controle
        Retorno: None
        '''
        if self.corpus_unico:
            print('Você está trabalhando com um corpus único.')
            return
        if dimensao in self.corpus:
            print('Essa dimensão já está incluída no controle.')
            return
        self.corpus.incluir(dimensao)
        if not 'Relacionamentos' in self.corpus: 
            self.corpus.incluir('Relacionamentos')

    def pesos_dimensoes(self):
        '''
        Mostra os pesos das dimensões.
        Retorno: None
        '''
        if self.corpus_unico:
            print('Você está trabalhando com um corpus único.')
            return
        print('Os pesos das dimensões são os seguintes:')
        for dimensao in self.dimensoes():
            print(f'--> {dimensao}: {self.corpus.peso(dimensao)}')

    def semelhantes(self, ficha=None, dimensoes=None, teste=False):
        '''
        Obtém as fichas mais semelhantes à ficha informada. Se corpus_unico for False, realiza a pesquisa nas dimensões
        informadas em dimensoes, sendo que, se dimensoes for None, realiza a pesquisa em todas as dimensoes. O resultado fica
        no atributo "resultados" que é um dicionário no qual o resultado de cada dimensão estará na chave com o nome da
        dimensão e o resultado final estará na chave "Final".
        Parâmetros:
            ficha (string) --> Nome da ficha que se deseja comparar
            dimensoes (list de string) --> Lista dos nomes das dimensões nas quais se deseja pesquisar. Se for None, pesquisa
                    em todas as dimensões
        Retorno: None
        '''
        self.resultados = {}
        if not ficha:
            print('Você tem que indicar uma ficha para analisar a semelhança.')
            return
        geral = ('geral', 'per_sim')
        # Em se tratando de um corpus único, apenas apresenta o resultado final da query
        if self.corpus_unico:
            ok, resultado = self.corpus.semelhantes(ficha=ficha, teste=teste)
            if not ok:
                print('Não foi possível obter o resultado da pesquisa')
                return
            # Inclui a informação da ordem das fichas
            resultado.reset_index(inplace=True)
            resultado.rename(columns={'index': 'Ficha'}, inplace=True)
            resultado.index = [i for i in range(1, resultado.shape[0]+1)]
            self.resultados['Detalhado'] = resultado
            self.resultados['Final'] = resultado[[('Ficha', ''), geral]]
            if self.max_resultados:
                 self.resultados['Detalhado'] = self.resultados['Detalhado'].head(self.max_resultados)
                 self.resultados['Final'] = self.resultados['Final'].head(self.max_resultados)
            return
        # Obtém a relação das dimensões se dimensoes for None
        if not dimensoes: dimensoes = self.dimensoes()
        # Monta o DataFrame com o resultado da query de cada dimensão
        # Se não for a primeira dimensão, junta com o resultado anterior
        primeira = True
        for dimensao in dimensoes:
            peso = self.corpus.peso(dimensao)
            ok, parcial = self.corpus[dimensao].semelhantes(ficha=ficha, teste=teste)
            if not ok:
                print(f'Não foi possível obter o resultado da pesquisa para a dimensão "{dimensao}"')
                self.resultados[dimensao] = None
                continue
            # Informa a ordem do resultado na dimensão, sem alterar o DataFrame original
            self.resultados[dimensao] = parcial.reset_index()
            self.resultados[dimensao].rename(columns={'index': 'Ficha'}, inplace=True)
            self.resultados[dimensao].index = [i for i in range(1, parcial.shape[0]+1)]
            if not teste and self.max_resultados:
                self.resultados[dimensao] = self.resultados[dimensao].head(self.max_resultados)
            if primeira:
                resultado = parcial.loc[:,[geral]]
                resultado[('geral', 'peso')] = peso
                resultado.columns = pd.MultiIndex.from_product([[dimensao], ['per_sim', 'peso']])
                primeira = False
            else:
                parcial = parcial.loc[:,[geral]]
                parcial[('geral', 'peso')] = peso
                parcial.columns = pd.MultiIndex.from_product([[dimensao], ['per_sim', 'peso']])
                resultado = resultado.join(parcial, how='outer')
        if primeira:
            print('Não foi possível obter o resultado da pesquisa para nenhuma dimensão')
            self.resultados['Final'] = None
            return
        # Preenches com zeros os valores não encontrados em cada dimensão
        resultado.fillna(0, inplace=True)
        # Calcula a probabilidade geral com base no peso de cada dimensão e ordena por esse valor em ordem decrescente
        resultado[geral] = [0 for i in range(resultado.shape[0])]
        peso_total = 0
        for dimensao in dimensoes:
            peso = self.corpus.peso(dimensao)
            resultado[geral] = resultado[geral] + (resultado[(dimensao, 'per_sim')] * resultado[(dimensao, 'peso')])
            peso_total += peso
        resultado[geral] = round(resultado[geral] / peso_total, 2)
        resultado.sort_values(by=geral, ascending=False, inplace=True)
        # Inclui a informação da ordem das fichas
        resultado.reset_index(inplace=True)
        resultado.rename(columns={'index': 'Ficha'}, inplace=True)
        resultado.index = [i for i in range(1, resultado.shape[0]+1)]
        self.resultados['Final'] = resultado
        if not teste and self.max_resultados:
            self.resultados['Final'] = self.resultados['Final'].head(self.max_resultados)

    def testar_dimensoes(self, dimensoes=None, vetor_testes=[], sucesso=100):
        '''
        Realiza testes nas dimensões, buscando a similaridade dos pares de teste.
        Parâmetros:
            dimensoes (list de str) --> Nomes das dimensoes que se deseja pesquisar. Se None, pesquisa todas (default: None)
            vetor_testes (list de tuple (str, str)) --> Lista de pares de fichas, sendo a primeira o argumento de pesquisa
                e a segunda a ficha cuja semelhança se espera encontrar
            sucesso (int) --> Posição máxima na qual pode estar a ficha cuja semelhança se deseja encontrar para ser considerado
                que o modelo obteve sucesso na pesquisa (default: 100)
        Retorno: um dicionário (str: DataFrame) onde as chaves são os nomes das dimensões e os valores são outros dicionários
                 com o resultado do teste para cada modelo na dimensão. Na chave "Geral" do dicionário principal há um DataFrame
                 com a quantidade de sucessos de todos as dimensoes.
        '''
        # Verifica se é um corpus único e se tem um vetor de testes
        if self.corpus_unico:
            print('Você está trabalhando com um corpus único')
            return
        if not vetor_testes:
            print('Você tem que indicar um vetor de testes para testar as dimensões.')
            return
        # Inicia as variáveis de trabalho
        dimensoes = dimensoes or self.dimensoes()
        testes_dimensoes = {}
        sucessos = []
        for dimensao in tqdm(dimensoes):
            if dimensao not in self.dimensoes():
                print(f'A dimensão "{dimensao}" não foi incluída no controle.')
                continue
            print(f'Testando a dimensão "{dimensao}".')
            testes_dimensoes[dimensao] = self.corpus[dimensao].testar_corpus(vetor_testes=vetor_testes, sucesso=sucesso)
            sucessos.append(testes_dimensoes[dimensao]['sucesso'].sum())
        testes_dimensoes['Geral'] = pd.DataFrame(data={'dimensões': dimensoes, 'sucessos': sucessos})
        return testes_dimensoes

    def testar_buscador(self, vetor_testes=[], sucesso=100):
        '''
        Realiza testes no buscador como um todo após ajustes dos pesos e dos modelos por meio da identificação da similaridade
        dos pares de teste.
        Parâmetros:
            vetor_testes (list de tuple (str, str)) --> Lista de pares de fichas, sendo a primeira o argumento de pesquisa
                e a segunda a ficha cuja semelhança se espera encontrar
            sucesso (int) --> Posição máxima na qual pode estar a ficha cuja semelhança se deseja encontrar para ser considerado
                que o modelo obteve sucesso na pesquisa (default: 100)
        Retorno: um dicionário (str: DataFrame) onde as chaves são os nomes das dimensões e os valores são outros dicionários
                 com o resultado do teste para cada modelo na dimensão. Na chave "Geral" do dicionário principal há um DataFrame
                 com a quantidade de sucessos de todos as dimensoes.
        '''
        # Inicia o dicionário que será usado para montar o DataFrame
        resultados = {'query': [], 'target': [], 'ordem': [], 'per_sim': [], 'sucesso': []}
        # Percorre o vetor de testes para realizar as pesquisas
        for query, target in tqdm(vetor_testes):
            resultados['query'].append(query)
            resultados['target'].append(target)
            self.semelhantes(ficha=query, teste=True)
            df = self.resultados['Final']
            if df is None: df = pd.DataFrame(data={'Ficha': [], ('geral', 'per_sim'): []})
            resultado = df[df['Ficha']==target][('geral', 'per_sim')]
            if resultado.empty:
                resultados['ordem'].append(None)
                resultados['per_sim'].append(None)
                resultados['sucesso'].append(None)
            else:
                per_sim = resultado.values[0]
                posicao = resultado.index[0]
                resultados['ordem'].append(posicao)
                resultados['per_sim'].append(per_sim)
                resultados['sucesso'].append(posicao <= sucesso)
        return pd.DataFrame(data=resultados)

    def _iniciar_twins(self):
        '''
        Inicializa os atributos da classe e faz os ajustes iniciais.
        Retorno: None
        '''
        est_base = ['projetos', obter_link_name(self.projeto)]
        base = '.'
        # Cria o caminho para a pasta do projeto
        for caminho in est_base:
            base = f'{base}/{caminho}'
            if not os.path.isdir(base): os.mkdir(base)
        # Verifica se já há o corpus definido
        self._arq_shelve = os.path.join(base, 'objetos.db')
        if os.path.isfile(f'{self._arq_shelve}.dat'):
            with shelve.open(self._arq_shelve) as db:
                twins = db['twins']
            # Recupera os valores anteriores dos parâmetros
            self.projeto = twins['projeto']
            self.max_resultados = twins['max_resultados']
            # Ajusta o atributo corpus_unico, se necessário
            if not twins['corpus_unico']:
                self.corpus = Dimensoes(self.projeto)
                if self.corpus_unico:
                    print(f'O projeto "{self.projeto}" já existe e é do tipo com dimensões.')
                    print('Foram carregadas as configurações do projeto já existente. ')
                    self.corpus_unico = False
            else:
                self.corpus = Corpus(nome=self.projeto, projeto=self.projeto)
                if not self.corpus_unico:
                    print(f'O projeto "{self.projeto}" já existe e é do tipo com um corpus único.')
                    print('Foram carregadas as configurações do projeto já existente. ')
                    self.corpus_unico = True
            # Atualiza o arquivo com a nova versão da classe
            self._salvar_twins()
            return
        # Se o controle é novo, inicia todos os valores e salva a configuração
        elif self.corpus_unico: self.corpus = Corpus(nome=self.projeto, projeto=self.projeto)
        else: self.corpus = Dimensoes(projeto=self.projeto)
        self._salvar_twins()

    def _salvar_twins(self):
        '''
        Persiste a situação atual do objeto do atributo corpus.
        Retorno: None
        '''
        twins = dict(corpus_unico = self.corpus_unico
                    ,projeto = self.projeto
                    ,max_resultados = self.max_resultados)
        with shelve.open(self._arq_shelve) as db:
            db['twins'] = twins
