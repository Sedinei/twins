# Imports Python
import re
import os
from tqdm.notebook import tqdm
import pandas as pd
import locale
import datetime as dt
import shelve
# Imports Gensim
#from gensim.corpora import MmCorpus
from gensim import utils as g_utils
# Imports Twins
from twins.utils import StreamCSV, TaggedCorpus, BOWCorpus, FormataDeltatime, obter_link_name
from twins.dao import DAOCorpus
from twins.models import Models

# Nomes internos simplificados para cronometrar o tempo de execução
AGORA = dt.datetime.now
TEMPO = FormataDeltatime()

# Seta localização
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# TODO
#   Análise de atributos:
#       * Montar gráficos para mostrar a distribuição dos tokens de um atributo pelas fichas
#       * Mostrar estatísticas dos tokens mais frequentes, distribuição, média, desvio padrão,...
#       * VER SE É REALMENTE NECESSÁRIO. ESPERAR PELAS ANÁLISES QUE TIVER QUE FAZER
#
#   Análise de distribuição de tokens por ficha
#       * Montar gráficos para mostrar a distribuição da quantidade de tokens distintos (tamanho do vetor) para cada ficha
#
#   Análise dos tokens mais importantes para a semelhança
#       * Mostrar quais "n" tokens que mais contribuiram para o percentual final de semelhança.

class Corpus:
    '''
    Representa um corpus, seu dicionário de palavras e suas respresentações vetorizadas. Incorpora ao corpus informações de
    relacionamentos identificados dos dados dos documentos de todos os CPFs e CNPJs encontrados nos corpus (atributos que
    contém cpf ou cnpj no nome). Os documentos são agrupados em fichas cujos nomes devem ser strings. A posição da ficha no
    corpus é a mesma da representação vetorizada dela, de modo a permitir a sua identificação nas rotinas de busca de fichas
    semelhantes.
    Parâmetros
        projeto (string) --> Nome do projeto ao qual pertence o corpus. Não é sensitive case, ignora acentos e quantidade de
                espaços entre as palavras
        nome (string) --> Nome do corpus. Não é sensitive case, ignora acentos e quantidade de espaços entre as palavras
                (default: "Geral")
    Atributos:
        nome (string) --> Nome do corpus (é sempre "Geral")
        projeto (string) --> Nome do projeto ao qual pertence o corpus
        acentos (list de string) --> Lista dos atributos, além dos "word", de cujos valores devem ser excluídos os acentos (default: lista vazia)
        tags_relac (list de string) --> Lista de palavras chaves que identificam relacionamentos em atributos (default: ['cpf', 'cnpj'])
        min_len (int) --> Tamanho mínimo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word" (default: 4)
        max_len (int) --> Tamanho máximo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word" (default: 100)
        no_below (int) --> Número mínimo de documentos no qual o token tem que aparecer para seguir no dicionário (default: 5)
        no_above (float) --> Percentual máximo de documentos no qual o token pode aparecer para seguir no diconário (default: 0.8)
        keep_n (int) --> Número máximo de tokens mais frequentes que seguirão no dicionário (default: 1000000)
        num_docs (int) --> Total de documentos lidos para montar o corpus (linhas de CSVs)
        num_atributos (int) --> Total de atributos constantes do corpus
        num_fichas (int) --> Total de fichas do corpus
        num_words (int) --> Total de palavras processadas no corpus
        num_tokens_full (int) --> Número de tokens no dicionário antes da filtragem
        freq_max_token_full (int) --> Número de documentos no qual aparece o token mais frequente antes da filtragem
        freq_min_token_full (int) --> Número de documentos no qual aparece o token menos frequente antes da filtragem
        num_fichas_token_full (int) --> Número total de fichas com ao menos um token antes da filtragem
        max_tokens_ficha_full (int) --> Máximo de tokens em uma única ficha antes da filtragem
        min_tokens_ficha_full (int) --> Mínimo de tokens em uma única ficha antes da filtragem
        avg_tokens_ficha_full (float) --> Média de tokens por ficha antes da filtragem
        sdv_tokens_ficha_full (float) --> Desvio padrão de tokens por ficha antes da filtragem
        num_tokens (int) --> Número de tokens após a filtragem
        freq_max_token (int) --> Número de documentos no qual aparece o token mais frequente após a filtragem
        freq_min_token (int) --> Número de documentos no qual aparece o token menos frequente após a filtragem
        num_fichas_token (int) --> Número total de fichas com ao menos um token após a filtragem
        max_tokens_ficha (int) --> Máximo de tokens em uma única ficha após a filtragem
        min_tokens_ficha (int) --> Mínimo de tokens em uma única ficha após a filtragem
        avg_tokens_ficha (float) --> Média de tokens por ficha após a filtragem
        sdv_tokens_ficha (float) --> Desvio padrão de tokens por ficha após a filtragem
    '''
    def __init__(self, projeto, nome='Geral'):
        # Atributos expostos do objeto que são persistidos
        self.nome = nome
        self.projeto = projeto
        self.acentos = []
        self.tags_relac = ['cpf', 'cnpj']
        self.min_len = 4
        self.max_len = 100
        self.no_below = 5
        self.no_above = 0.8
        self.keep_n = 1000000
        self.num_docs = 0
        self.num_atributos = 0
        self.num_fichas = 0
        self.num_words = 0
        self.num_tokens_full = 0
        self.freq_max_token_full = 0
        self.freq_min_token_full = 0
        self.num_fichas_token_full = 0
        self.max_tokens_ficha_full = 0
        self.min_tokens_ficha_full = 0
        self.avg_tokens_ficha_full = 0
        self.sdv_tokens_ficha_full = 0
        self.num_tokens = 0
        self.freq_max_token = 0
        self.freq_min_token = 0
        self.num_fichas_token = 0
        self.max_tokens_ficha = 0
        self.min_tokens_ficha = 0
        self.avg_tokens_ficha = 0
        self.sdv_tokens_ficha = 0
        # Atributos expostos do objeto que NÃO são persistidos
        self.modelos = None
        # Atributos internos que são persistidos
        self._atributo_ficha = ''
        self._lendo_csv = False
        self._arquivo_csv = ''
        self._sep = None
        self._nrows = None
        self._total_docs = None
        self._docs_lidos = 0
        self._id_origem = None
        self._atributos = {}
        self._has_dict = False
        # Atributos internos que não são persistidos
        self._pastas = {}
        self._regex = {}
        self._link_nome = None
        self._shelf = None
        self._arqs = {}
        self._dao = None
        self._tokens = {}
        self._update_relac = False  # Atributo de controle para a subclasse CorpusDimensao
        # Obtém as configurações anteriores do corpus ou inicia os arquivos e nomes de arquivos
        self._iniciar_corpus()

    def ajustar_acentos(self, atributos, incluir=True):
        '''
        Exclui ou inclui atributos da lista dos cujos valores devem ter os acentos retirados.
        Parâmetros:
            atributos (list de string) --> Lista de atributos a serem incluídos ou retirados da relação cujos valores devem
                    ter seus acentos retirados
            incluir (boolean) --> Indica se os atributos devem ser incluídos (True) ou retirados (False) da relação cujos valores devem
                    ter seus acentos retirados (default: True)
        Retorno: None
        '''
        # Se já tem um dicionário montado para a dimensão, não altera os parâmetros
        if self._has_dict:
            print(f'Há um dicionário montado no corpus "{self.nome}" com os parâmetros anteriores.')
            print('Os novos parâmetros serão aplicadas apenas para as próximas entradas de dados.')
        if incluir:
            # Inclui os novos atributos cujos valores devem ter os acentos removidos
            self.acentos += atributos
        else:
            # Remove da lista os atributos cujos valores tem que ter os acentos removidos
            for atributo in atributos:
                self.acentos.remove(atributo)
        # Persiste a nova relação de atributos cujos acentos devem ser removidos
        self._salvar_configuracoes()
        
    def ajustar_dicionario(self, no_below=None, no_above=None, keep_n=None):
        '''
        Altera os hiperparâmetros do corpus referentes ao dicionário. Os hiperparâmetros não informados no método manterão os
        valores definido anteriormente. Se já existir um dicionário do corpus, altera-o com base nos novos parâmetros.
        Parâmetros:
            no_below (int) --> Número mínimo de documentos no qual o token tem que aparecer para seguir no dicionário
            no_above (float) --> Percentual máximo de documentos no qual o token pode aparecer para seguir no diconário
            keep_n (int) --> Número máximo de tokens mais frequentes que seguirão no dicionário
        Retorno: None
        '''
        # Realiza as alterações nos hiperparâmetros
        if no_below: self.no_below = no_below
        if no_above: self.no_above = no_above
        if keep_n: self.keep_n = keep_n
        # Persiste os novos parâmetros
        self._salvar_configuracoes()

    def ajustar_tags_relacionamentos(self, tags, incluir=True):
        '''
        Exclui ou inclui tags para identificação de relacionamentos.
        Parâmetros:
            tags (list de string) --> Lista de tags a serem incluídos ou retirados da relação das tags para identificação
                    de relacionamentos
            incluir (boolean) --> Indica se as tags devem ser incluídas (True) ou retiradas (False) da relação das tags
                    para identificação de relacionamentos
        Retorno: None
        '''
        if incluir:
            # Inclui as novas tags para identificação de relacionamentos
            self.tags_relac += tags
        else:
            # Remove da relação as tags para identificação de relacionamentos
            for tag in tags:
                self.tags_relac.remove(tag)
        # Persiste a nova relação de tags
        self._salvar_configuracoes()

    def ajustar_words(self, min_len=None, max_len=None):
        '''
        Altera os hiperparâmetros do corpus referentes aos atributos word. Os hiperparâmetros não informados no método manterão
        os seus valores anteriormente definidos. Esses novos parâmetros são aplicados apenas sobre novas entradas de dados, não
        afetando os tokens de atributos word que já existam no corpus.
        Parâmetros:
            min_len (int) --> Tamanho mínimo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word"
            max_len (int) --> Tamanho máximo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word"
        Retorno: None
        '''
        # Se já tem um dicionário montado para a dimensão, não altera os parâmetros
        if self._has_dict:
            print(f'Há um dicionário montado no corpus "{self.nome}" com os parâmetros anteriores.')
            print('Os novos parâmetros serão aplicadas apenas para as próximas entradas de dados.')
        # Realiza as alterações nos hiperparâmetros
        if min_len: self.min_len = min_len
        if max_len: self.max_len = max_len
        # Persiste os novos parâmetros
        self._salvar_configuracoes()

    def atributos(self):
        '''
        Retorna a lista de atributos que fazem parte do corpus.
        Retorno: uma lista de atributos (list de strings)
        '''
        return self._dao.obter_atributos()

    def corpus(self, tipo='bow', ficha=None):
        '''
        Retorna o streamming do corpus no tipo indicado ou um documento no tipo indicado referente à ficha passada.
        Parâmetros:
            tipo (string) --> Formato de representação do corpus. Os valores possíveis são 'bow', 'tfidf', 'tfidf_pivot',
                        'lsi', 'lda', 'tagged' e 'doc2vec' (default: 'bow')
            ficha (string) --> Nome da ficha cuja representação deseja receber no tipo indicado. Se for None, retorna o
                        streamming do corpus (default: None)
        Retorno:
            * um streamming da representação da ficha no tipo indicado, se ficha não for None
            * None, se for indicada uma ficha que não faz parte do corpus ou não houver um dicionário montado
        '''
        # Verifica se já há um dicionário montado
        if not self._has_dict:
            print(f'O corpus "{self.nome}" ainda não tem um dicionário montado.')
            return
        if tipo not in self.modelos and tipo not in ['bow', 'tagged']:
            print(f'O tipo "{tipo}" não foi implementado.')
        # Obtém o streamming do corpus no tipo pedido
        if tipo == 'bow': base = BOWCorpus(self)
        elif tipo == 'tagged': base = TaggedCorpus(self)
        elif tipo == 'doc2vec':
            model = self.modelos[tipo]
            base = TaggedCorpus(self)
        elif tipo == 'lsi':
            model_tfidf = self.modelos['tfidf']
            if not model_tfidf:
                print(f'Não há ainda o corpus no tipo "tfidf" montado, que é necessário para montar o corpus no tipo "lsi".')
                return
            model = self.modelos[tipo]
        else:
            model = self.modelos[tipo]
            base = BOWCorpus(self)
        if tipo not in ['bow', 'tagged'] and not model:
            print(f'Não há ainda o corpus no tipo "{tipo}" montado.')
            return
        # Retorna o streamming se ficha é None
        if not ficha:
            if tipo == 'lsi': return model[model_tfidf[BOWCorpus(self)]]
            elif tipo in ['bow', 'tagged']: return base
            else: return model[base]
        # Verifica se a ficha faz parte do corpus
        id_ficha = self.ficha2id(ficha)
        if not id_ficha:
            print(f'A ficha "{ficha}" não faz parte do corpus "{self.nome}".')
            return None
        # Retorna a representação da ficha no tipo pedido
        if tipo == 'bow': return base[id_ficha]
        elif tipo == 'lsi': return model[model_tfidf[base[id_ficha]]]
        else: return model[base[id_ficha]]

    def dicionario(self):
        '''
        Retorna um dict a partir do dicionário do corpus onde a chave é o id do token e o valor é o token.
        Retorno: dict onde a chave é o id do token (int) e o valor é o token (string)
        '''
        # Verifica se já há um dicionário montado
        if not self._has_dict:
            print(f'O corpus "{self.nome}" ainda não tem um dicionário montado.')
            return
        return self._dao.obter_tokens_dicionario()

    def fichas(self):
        '''
        Retorna a lista de fichas que fazem parte do corpus. Essa lista está na mesma ordem dos documentos das fichas nos corpus.
        Retorno: uma lista de fichas (list de strings)
        '''
        return self._dao.obter_fichas()

    def ficha2id(self, ficha):
        '''
        Retorna o índice do documento ao qual se refere a ficha passada em uma representação vetorial. Se a ficha não estiver
        presente  no corpus, retorna None.
        Parâmetros:
            ficha (string) --> A ficha cujo id se deseja obter
        Retorno: o id da ficha passada (int)
        '''
        id_ficha = self._dao.obter_id_ficha(ficha)
        if id_ficha is None:
            print(f'A ficha "{ficha}" não faz parte do corpus "{self.nome}".')
            return None
        return id_ficha

    def id2ficha(self, id_ficha):
        '''
        Retorna o nome da ficha associada à posição de um documento em um corpus.
        Parâmetros:
            id_ficha (int) --> Número da posição de um documento em um corpus base zero
        Retorno: o nome da ficha que corresponde à posição passada (string) ou None se não encontrar essa posição no corpus.
        '''
        ficha = self._dao.obter_ficha_id(id_ficha)
        if ficha is None:
            print(f'A posição "{id_ficha}" está fora do limite de posições de fichas do corpus "{self.nome}".')
            return None
        return ficha

    def incluir_documentos_csv(self, arq_csv=None, sep=',', nrows=None, total_docs=None, ind_tokens=True):
        '''
        Povoa um corpus a partir de um arquivo CSV. Nesse arquivo, a primeira coluna é o nome da ficha no corpus
        enquanto as demais são os valores dos atributos do documento. O nome da ficha é um string e o nome do atributo é o nome
        da coluna, enquanto que os valores do atributo são os valores da célula separados por espaço. Se os pares atributo/valor
        estiverem previamente toquenizados, "ind_tokens" deve ser True. Caso contrário, "ind_tokens" deve ser False e os
        tokens serão formados pela concatenação do nome da coluna, do separador '_' e de cada valor do atributo. Os atributos que
        tiverem "word" no nome não devem estar tokenizados previamente, pois sofrem um pré-processamento diferenciado após feito
        o split do valor da célula do atributo.
        Parâmetros:
            arq_csv (string) --> O endereço completo onde se encontra o arquivo CSV com os dados para a formação do corpus. Se None,
                             apenas verifica se há um processo de leitura anterior que foi interrompido e dá seguimento a ele.
            sep (string) --> Separador usado no arquivo CSV (default: ',')
            nrows (int) --> Número máximo de linhas a serem lidas do arquivo. Se None, lê todo o arquivo (default: None)
            total_docs (int) --> Número total de documentos a serem lidos que será considerado para informação na barra de
                                 progresso. Se None, a barra apenas conta o número de documentos lidos (default: None)
            ind_tokens (boolean) --> Indica se os valores já são tokens montados (True) ou se é preciso compor os tokens agrupando
                                     os valores das linhas com os nomes das colunas (False) (default: True)
        Retorno: None
        '''
        # Verifica se a dimensão é a relacionamentos não é derivada de outra
        if self._link_nome == 'relacionamentos':
            print('Você não deve incluir dados diretamente no corpus da dimensão "Relacionamentos".')
            return
        # Verifica se não havia um processo de leitura anterior
        if self._lendo_csv:
            print(f'Há um processo de leitura do arquivo "{self._arquivo_csv}" que ainda não foi concluído.')
            print(f'Já foram processados {self._docs_lidos} documentos.')
            print(f'Será dado seguimento a esse processo de leitura.')
            if arq_csv and arq_csv != self._arquivo_csv:
                print(f'Ao final da leitura, execute novamente este método para ler o arquivo "{arq_csv}".')
        else:
            # Seta as configurações de leitura
            self._lendo_csv = True
            self._arquivo_csv = arq_csv
            self._sep = sep
            self._nrows = nrows
            if (total_docs and nrows) and total_docs <= nrows: self._total_docs = total_docs
            elif (total_docs and nrows) and total_docs > nrows: self._total_docs = nrows
            elif not total_docs and nrows: self._total_docs = nrows
            else: self._total_docs = total_docs
            self._docs_lidos = 0
        # Cria o streamming dos dados do documento
        reader = StreamCSV(self._arquivo_csv, sep=self._sep, nrows=self._nrows, start=self._docs_lidos)
        if self._docs_lidos == 0:
            # Faz os registros iniciais, já que não começou a leitura do CSV
            self._id_origem = self._dao.registrar_origem(self._arquivo_csv, self._nrows)
            self._atributo_ficha = reader.atributos[0]
            self._atributos = self._dao.registrar_lista_atributos(reader.atributos[1:])
            self._salvar_configuracoes()
        # Inclui no corpus os dados do CSV
        if self._total_docs: total = self._total_docs - self._docs_lidos
        for chunk in tqdm(reader, desc='Reading CSV:', total=total):
            self._montar_corpus(chunk, ind_tokens)
        # Anota o final da leitura e a quantidade atual de documentos lidos
        self.num_docs += self._docs_lidos
        self._lendo_csv = False
        self._salvar_configuracoes()
        # Realizar o encerramento do método
        self._encerrar_incluir_documentos()

    def infos(self):
        '''
        Mostra as estatísticas do corpus.
        Retorno: None
        '''
        # Calcula os percentuais de fichas que estão no token mais frequente
        if not self.num_fichas: full_perc = perc = 0
        else:
            full_perc = (self.freq_max_token_full/self.num_fichas)*100
            perc = (self.freq_max_token/self.num_fichas)*100
        # Formata os números que aparecerão no informativo
        num_fichas = locale.format('%d', self.num_fichas, grouping=True)
        num_atributos = locale.format('%d', self.num_atributos, grouping=True)
        num_docs = locale.format('%d', self.num_docs, grouping=True)
        num_words = locale.format('%d', self.num_words, grouping=True)
        min_len = locale.format('%d', self.min_len, grouping=True)
        max_len = locale.format('%d', self.max_len, grouping=True)
        num_acentos = locale.format('%d', len(self.acentos), grouping=True)
        no_below = locale.format('%d', self.no_below, grouping=True)
        no_above = locale.format('%.02f', self.no_above*100, grouping=True)
        keep_n = locale.format('%d', self.keep_n, grouping=True)
        num_tokens_full = locale.format('%d', self.num_tokens_full, grouping=True)
        freq_max_token_full = locale.format('%d', self.freq_max_token_full, grouping=True)
        full_perc = locale.format('%.02f', full_perc, grouping=True)
        freq_min_token_full = locale.format('%d', self.freq_min_token_full, grouping=True)
        num_fichas_token_full = locale.format('%d', self.num_fichas_token_full, grouping=True)
        max_tokens_ficha_full = locale.format('%d', self.max_tokens_ficha_full, grouping=True)
        min_tokens_ficha_full = locale.format('%d', self.min_tokens_ficha_full, grouping=True)
        avg_tokens_ficha_full = locale.format('%.02f', self.avg_tokens_ficha_full, grouping=True)
        sdv_tokens_ficha_full = locale.format('%.02f', self.sdv_tokens_ficha_full, grouping=True)
        num_tokens = locale.format('%d', self.num_tokens, grouping=True)
        freq_max_token = locale.format('%d', self.freq_max_token, grouping=True)
        perc = locale.format('%.02f', perc, grouping=True)
        freq_min_token = locale.format('%d', self.freq_min_token, grouping=True)
        num_fichas_token = locale.format('%d', self.num_fichas_token, grouping=True)
        max_tokens_ficha = locale.format('%d', self.max_tokens_ficha, grouping=True)
        min_tokens_ficha = locale.format('%d', self.min_tokens_ficha, grouping=True)
        avg_tokens_ficha = locale.format('%.02f', self.avg_tokens_ficha, grouping=True)
        sdv_tokens_ficha = locale.format('%.02f', self.sdv_tokens_ficha, grouping=True)
        # Monta o informativo
        print(f'Estatísticas do corpus: "{self.nome}"')
        print()
        print(f'Total de fichas: {num_fichas:<40}Total de atributos: {num_atributos}')
        print(f'Total de documentos lidos: {num_docs:<30}Total de palavras processadas: {num_words}')
        print()
        print('-->Parâmetros de filtragem para inclusão de um atributo word:')
        print(f'Mínimo de caracteres: {min_len:<35}Máximo de caracteres: {max_len}')
        print(f'Número de atributos com remoção de acentuação: {num_acentos}')
        print()
        print('-->Parâmetros de filtragem de cada token no dicionário do corpus:')
        print(f'Número mínimo de fichas: {no_below:<26}Percentual máximo de fichas: {no_above}%')
        print(f'Número máximo de tokens: {keep_n}')
        print()
        print('-->Estatísticas do dicionário ANTES da filtragem:')
        print(f'Número total de tokens: {num_tokens_full}')
        print(f'Número de fichas do token mais frequente: {freq_max_token_full} ({full_perc}%)')
        print(f'Número de fichas do token menos frequente: {freq_min_token_full}')
        print(f'Número total de fichas com ao menos um token: {num_fichas_token_full}')
        print(f'Máximo de tokens em uma única ficha: {max_tokens_ficha_full}')
        print(f'Mínimo de tokens em uma única ficha: {min_tokens_ficha_full}')
        print(f'Média de tokens para cada ficha: {avg_tokens_ficha_full}')
        print(f'Desvio padrão de tokens para cada ficha: {sdv_tokens_ficha_full}')
        print()
        print('-->Estatísticas do dicionário APÓS a filtragem:')
        print(f'Número total de tokens: {num_tokens}')
        print(f'Número de documentos do token mais frequente: {freq_max_token} ({perc}%)')
        print(f'Número de documentos do token menos frequente: {freq_min_token}')
        print(f'Número total de fichas com ao menos um token: {num_fichas_token}')
        print(f'Máximo de tokens em uma única ficha: {max_tokens_ficha}')
        print(f'Mínimo de tokens em uma única ficha: {min_tokens_ficha}')
        print(f'Média de tokens para cada ficha: {avg_tokens_ficha}')
        print(f'Desvio padrão de tokens para cada ficha: {sdv_tokens_ficha}')

    def montar_dicionario(self):
        '''
        Monta o dicionário a partir do corpus, obtém as estatísticas e vetoriza o corpus nos tipos implementados.
        Retorno: None
        '''
        # Monta o dicionário filtrado e recebe as estatísticas
        print(f'Montando o dicionário do corpus "{self.nome}"')
        t0 = AGORA()
        est = self._dao.montar_dicionario(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)
        TEMPO.formatar(AGORA() - t0)
        print(f'O dicionário foi montado em {TEMPO}')
        # Informa que há um dicionário
        if not self._has_dict:
            self._has_dict = True
            self._salvar_configuracoes()
        # Salva as estatísticas
        self._povoar_atributos(est, zero=True)

    def parametros_modelos(self):
        '''
        Mostra os parâmetros usados nos modelos do corpus
        Retorno: None
        '''
        dados = self.modelos.parametros()
        print(f'Parâmetros usados nos modelos do corpus "{self.nome}" são os seguintes.')
        for modelo, parametros in dados.items():
            print(f'  * Modelos "{modelo}":')
            for parametro, valor in parametros.items():
                print(f'    --> {parametro} = {valor}')

    def semelhantes(self, ficha, teste=False):
        '''
        Pesquisa no corpus quais fichas tem características mais semelhantes às da ficha indicada.
        Parâmetros:
            ficha (string) --> Ficha que servirá de comparação para buscar as semelhantes
        Retorno: uma tupla onde o primeiro elemento é um indicador de que há um resultado e o segundo é um Pandas DataFrame
            na ordem decrescente de semelhança das fichas
        '''
        id_ficha = self.ficha2id(ficha)
        if id_ficha is None: return False, None
        resultado = self.modelos.semelhantes(id_ficha, teste=teste)
        # Substitui os id's pelos valores das fichas
        fichas = []
        for indice in resultado.index:
            fichas.append(self.id2ficha(indice))
        resultado.index = fichas
        return True, resultado

    def testar_corpus(self, vetor_testes=[], sucesso=100):
        '''
        Realiza testes no corpus, buscando a similaridade dos pares de teste.
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
            ok, df = self.semelhantes(query)
            if not ok or target not in df.index:
                resultados['ordem'].append(None)
                resultados['per_sim'].append(None)
                resultados['sucesso'].append(None)
            else:
                per_sim, posicao = df.loc[target, [('geral', 'per_sim'), ('geral', 'ordem')]]
                resultados['ordem'].append(posicao)
                resultados['per_sim'].append(per_sim)
                resultados['sucesso'].append(posicao <= sucesso)
        return pd.DataFrame(data=resultados)

    def testar_modelos_topicos(self, modelos=None, num_topicos=[20, 50, 100, 200, 300, 400, 500, 1000, 1500]
                              ,perc_fichas=0.2, vetor_testes=[], tipo_teste='similaridade'):
        '''
        Testa os modelos gerados por tópicos para uma lista de quantidade de tópicos a fim de encontrar
        o melhor número de tópicos para o modelo com relação ao corpus.
        Parâmetros:
            modelos (list de strings) --> Modelos a serem testados: "lda", "lsi" ou "doc2vec". Se None, testa todos.
            num_topics (list de int) --> Lista de números de tópicos a serem testados (default: [20, 50, 100, 200, 300, 400, 500, 1000, 1500])
            per_fichas (float) --> Percentual de fichas do corpus a serem considerados para o teste (default: 0.2)
            vetor_testes (list de tuple (str, str)) --> Lista de pares de fichas, sendo a primeira o argumento de pesquisa
                e a segunda a ficha cuja semelhança se espera encontrar
            tipo_testes (string) --> Tipo de teste: "u_mass" ou "similaridade" (default: "similaridade")
        Retorno: um dicionário de dicionários. A chave do dicionário principal é o nome do modelo. Para cada modelo há outro
            dicionário, com as seguintes chaves:
                "medida" --> Um objeto Series de pandas com os valores de medida calculados para o modelo com cada número
                             de tópicos (os números de tópicos são os índices).
                "modelos" --> Um dicionário onde a chave é o número de tópicos e o valor é o modelo gerado para aquele número
                              de tópicos.
        '''
        # Verifica se consegue carregar no objeto de modelos o dicionário do corpus
        if not self._has_dict:
            print('Não há um dicionário do corpus montado.')
            print('Não é possível testar os modelos de tópicos.')
            return
        if not vetor_testes:
            print('Você precisa indicar um vetor de testes.')
            return
        # Monta os corpus e matrizes de similaridade nos tipos implementados
        testes_modelos = {}
        tipos_modelos = ['lda', 'lsi', 'doc2vec']
        modelos = modelos or tipos_modelos
        for modelo in tqdm(modelos):
            if modelo not in tipos_modelos:
                print(f'O modelo "{modelo}" ou não foi implementado ou não é de tópicos.')
            print(f'Testando o modelo para tranformar o corpus "{self.nome}" no tipo "{modelo}"')
            t0 = AGORA()
            testes = self.modelos.testar_num_topics(modelo=modelo, num_topicos=num_topicos, perc_fichas=perc_fichas
                                                   ,vetor_testes=vetor_testes, tipo_teste=tipo_teste)
            TEMPO.formatar(AGORA() - t0)
            print(f'Modelo "{modelo}" testado em {TEMPO}')
            print()
            if not testes:
                testes_modelos[modelo] = None
                continue
            medida = pd.Series([testes[num]['medida'] for num in testes], index=list(testes.keys()))
            medida = medida.sort_index()
            testes_modelos[modelo] = {'medida': medida, 'modelos': {num: testes[num]['modelo'] for num in testes}}
        return testes_modelos

    def testar_modelos(self, modelos=None, vetor_testes=[], sucesso=100):
        '''
        Realiza testes nos modelos, buscando a similaridade dos pares de teste.
        Parâmetros:
            modelos (list de str) --> Nomes dos modelos que se deseja pesquisar. Se None, pesquisa todos (default: None)
            vetor_testes (list de tuple (str, str)) --> Lista de pares de fichas, sendo a primeira o argumento de pesquisa
                e a segunda a ficha cuja semelhança se espera encontrar
            sucesso (int) --> Posição máxima na qual pode estar a ficha cuja semelhança se deseja encontrar para ser considerado
                que o modelo obteve sucesso na pesquisa (default: 100)
        Retorno: um dicionário (str: DataFrame) onde as chaves são os nomes dos modelos e os valores são os DataFrames com os
                 resultados do teste. Na chave "Geral" há um DataFrame com a quantidade de sucessos de todos os modelos.
        '''
        # Verifica se há um dicionário do corpus montado
        if not self._has_dict:
            print('Não há um dicionário do corpus montado.')
            print('Não é possível testar os modelos.')
            return
        if not vetor_testes:
            print('Você precisa indicar um vetor de testes.')
            return
        # Monta os corpus e matrizes de similaridade nos tipos implementados
        testes_modelos = {}
        sucessos = []
        modelos = modelos or self.modelos.tipos_modelos()
        for modelo in tqdm(modelos):
            if modelo not in self.modelos:
                print(f'O modelo "{modelo}" ou não foi implementado.')
            print(f'Testando o modelo para tranformar o corpus "{self.nome}" no tipo "{modelo}"')
            t0 = AGORA()
            testes_modelos[modelo] = self.modelos.testar_modelo(modelo=modelo, vetor_testes=vetor_testes, sucesso=sucesso)
            sucessos.append(testes_modelos[modelo]['sucesso'].sum())
            TEMPO.formatar(AGORA() - t0)
            print(f'Modelo "{modelo}" testado em {TEMPO}')
            print()
        testes_modelos['Geral'] = pd.DataFrame(data={'modelo': modelos, 'sucessos': sucessos})
        return testes_modelos

    def vetorizar(self, modelos=None):
        '''
        Cria modelos para representações vetorizadas do corpus.
        Parâmetros:
            modelos (list de string) --> Lista de modelos a serem executados. Se None, executa todos os modelos.
        Retorno: None
        '''
        # Verifica se consegue carregar no objeto de modelos o dicionário do corpus
        dicionario = self.dicionario()
        if not dicionario:
            print('Não foi possível carregar o dicionário do corpus')
            print('Não serão montadas as demais representações vetoriais do corpus')
            return
        # Monta os corpus e matrizes de similaridade nos tipos implementados
        if not modelos: modelos = self.modelos.tipos_modelos()
        for modelo in tqdm(modelos):
            # Verifica se o modelo já foi implementado
            if modelo not in self.modelos:
                print(f'O modelo "{modelo}" não foi implementado.')
                continue
            print(f'Gerando o modelo para tranformar o corpus "{self.nome}" no tipo "{modelo}"')
            t0 = AGORA()
            self.modelos.gerar_modelo(modelo=modelo)
            TEMPO.formatar(AGORA() - t0)
            print(f'Corpus no tipo "{modelo}" montado em {TEMPO}')

    def _agregar_tokens(self, tokens, id_atributo):
        '''
        Monta o dicionário de tokens, agregando as ocorrências de cada token e obtendo o seu id.
        Parâmetros:
            tokens (list de string) --> Relação de tokens a serem agregados
            id_atributo (int) --> Id do atributo ao qual pertence o token
        '''
        for token in tokens:
            # Verifica se o token já se encontra no dicionário
            if not self._tokens.get(token):
                # Obtém o id do token e o registra no dicionário
                self._tokens[token] = {'id_atributo': id_atributo, 'id_token': self._dao.registrar_token(token), 'freq_token': 1}
            # Se já existe no dicionário, acrescenta no contador de frequência
            else: self._tokens[token]['freq_token'] += 1

    def _condicao_ok(self):
        '''
        Método que verifica a condição necessária para a execução do método.
        Esse método foi criado para poder ser sobrescrito na subclasse CorpusDimensao.
        Retorno: True
        '''
        return True

    def _encerrar_incluir_documentos(self):
        '''
        Encerra o método incluir_documentos_csv.
        Esse método foi criado para poder ser sobrescrito na subclasse CorpusDimensao.
        Retorno: None
        '''
        # Obtém os dados do corpus
        self._obter_dados_corpus()
        # Monta o dicionário
        self.montar_dicionario()
        # Cria as versões vetorizadas do corpus
        self.vetorizar()

    def _iniciar_corpus(self):
        '''
        Verifica se existe o DB do corpus, iniciando-o se não existir e persistindo as configurações default. Se existir,
        recupera as configurações do objeto. Também seta os caminhos dos diversos arquivos do sistema e instancia as
        classes que são utilizadas.
        Retorno: None
        '''
        # Obtém o nome para link
        self._link_nome = obter_link_name(self.nome)
        self._shelf = f'corpus_{self._link_nome}'
        # Define o nome da pasta de trabalho e a cria, se não existir
        pastas = ['corpus', 'modelos', 'indices']
        est_base = ['projetos', obter_link_name(self.projeto)]
        base = '.'
        # Cria o caminho para a pasta do projeto
        for caminho in est_base:
            base = f'{base}/{caminho}'
            if not os.path.isdir(base): os.mkdir(base)
        self._pastas['projeto'] = base
        # Cria as pastas dos arquivos do projeto, se não existir
        for nome_pasta in pastas:
            pasta = f'{base}/{nome_pasta}'
            if not os.path.isdir(pasta): os.mkdir(pasta)
            self._pastas[nome_pasta] = pasta
        # Define os nomes do arquivo com o DB do corpus e com o Shelve
        self._arqs['db'] = os.path.join(self._pastas['corpus'], f'{self._link_nome}.db')
        self._arqs['shelve'] = os.path.join(self._pastas['projeto'], 'objetos.db')
        # Instancia a classe DAOCorpus e a inicia para criar as tabelas do banco, se for o caso
        self._dao = DAOCorpus(self._arqs['db'])
        self._dao.iniciar_dao()
        # Verifica se já há dados de configurações do corpus
        if os.path.isfile(f'{self._arqs["shelve"]}.dat'):
            # Verifica se os dados do corpus estão no arquivo shelve
            with shelve.open(self._arqs['shelve']) as db:
                if self._shelf not in db: config = None
                else: config = db[self._shelf]
            if config: self._povoar_atributos(config)
        # Compila os regex que serão usados no corpus
        self._regex['word'] = re.compile(r'(_|\b)word(_|\b)')
        self._regex['espaco'] = re.compile(r'\s+')
        self._regex['carac'] = re.compile(r'[^a-z\s]')
        for tag in self.tags_relac:
            self._regex[f'relac_{tag}'] = re.compile(r'(_|\b){}(_|\b)'.format(tag))
        # Instancia um objeto Models para o corpus
        self.modelos = Models(self)
        # Recupera os dados anteriores e atualiza o arquivo com a nova versão da classe
        self._salvar_configuracoes()

    def _qdb(self, sql, t=None):
        '''
        Método para realização de consultas SQL genéricas no DB.
        Parâmetros:
            sql (string) --> Código SQL a ser consultado com ou sem "?"
            t (tuple) --> Se o SQL tiver "?", são os valores que deverão ser substituídos no código
        Retorno: lista de tuplas, sendo que cada tupla representa uma linha do resultado da consulta
        '''
        return self._dao.consultar_db(sql, t)

    def _montar_corpus(self, chunk, ind_tokens):
        '''
        Percorre as colunas do objeto zip, quebrando os valores de cada atributo e formando os tokens para fins de geração
        do corpus.
        Parâmetros:
            chunk (zip) --> Um objeto zip que contém tuplas nas quais no primeiro elemento está o atributo e no segundo o valor
            ind_tokens (boolean) --> Indica se os valores já são tokens montados (True) ou se é preciso compor os tokens agrupando
                                     os valores das linhas com os nomes das colunas (False)
        Retorno: None
        '''
        self._tokens = {}
        # Itera os campos da linha para criar os tokens do documento
        for atributo, values in chunk:
            # Obtém o valor da ficha
            if atributo == self._atributo_ficha:
                # Registra a ficha e obtém o seu id
                ficha = values
                id_ficha = self._dao.registrar_ficha(ficha)
                continue
            # Se não tem valor a ser computado no atributo, passa para o próximo
            if not values: continue
            # Verifica se é um atributo word para realizar o pré-processamento diferenciado
            if self._regex['word'].search(atributo):
                self._agregar_tokens(self._obter_token_word(values, atributo), self._atributos[atributo])
                continue
            # Verifica se é um atributo que precisa remover a acentuação
            if atributo in self.acentos: values = g_utils.deaccent(values)
            # Se o atributo não é word, apenas separa os valores e agrega os tokens
            if ind_tokens: tokens = values.split()
            else: tokens = [f'{atributo}_{value}' for value in values.split()]
            self._agregar_tokens(tokens, self._atributos[atributo])
            # Verifica se é um atributo de relacionamento e inclui valores genéricos de relacionamentos no corpus
            for tag in self.tags_relac:
                if not self._regex[f'relac_{tag}'].search(atributo): continue
                new_atributo = tag
                retorno = self._tratar_relacionamentos(tokens, atributo, new_atributo)
                # Em CorpusDimensao a função _tratar_relacionamentos retorna None, pois os relacioamentos ficam em outro corpus
                if retorno:
                    # Verifica se o atributo do tipo de pessoa já está registrado no corpus
                    if not self._atributos.get(new_atributo):
                        self._atributos[new_atributo] = self._dao.registrar_atributo(new_atributo)
                    self._agregar_tokens(retorno, self._atributos[atributo])
        # Monta a lista de ocorrências a serem registradas
        ocorrencias = [(self._id_origem, id_ficha, self._tokens[token]['id_atributo'], self._tokens[token]['id_token'],
                       self._tokens[token]['freq_token']) for token in self._tokens]
        # Regista na base do corpus as ocorrências
        self._dao.registrar_frequencias(ocorrencias)
        # Verifica se há relacionamentos a registrar ==> PARA A SUBCLASSE Corpus_dimensao
        if self._update_relac: self._salvar_relacionamentos(ficha)
        # Incrementa o contador de documentos lidos e salva-o
        self._docs_lidos += 1
        self._salvar_configuracoes()

    def _obter_dados_corpus(self):
        '''
        Povoa os dados do corpus após uma importação. É um método separado em razão do CorpusDimensao
        Retorno: None
        '''
        # Obtém os dados do corpus
        print(f'Obtendo os dados do corpus "{self.nome}"')
        t0 = AGORA()
        self._povoar_atributos(self._dao.obter_dados_corpus(), zero=True)
        TEMPO.formatar(AGORA() - t0)
        print(f'Obteve os dados do corpus em {TEMPO}')

    def _obter_token_word(self, values, atributo):
        '''
        Realiza o pré-processamento dos atributos word e tokeniza os valores. O pré-processamento consiste nas seguintes
        etapas, na sequência (a etapa seguinte é realizada sobre o resultado da anterior):
            1) Passa todas as palavras para minúsculas;
            2) Retira qualquer caracter que não seja alfabético;
            3) Mantém apenas um espaço entre cada palavra;
            4) Retira as acentuações;
            5) Retira qualquer palavra com comprimento menor que min_len;
            6) Retira qualquer palavra com comprimento maior que max_len;
            7) Monta uma lista com as palavras resultantes separando-as pelo espaço;
            8) Monta o token pela concatenação do nome do atributo, "_" e a palavra.
        Parâmetros:
            values (string) --> Cadeia de palavras separadas por espaço a serem tokenizadas
            atributo (string) --> Atributo que servirá de base para a tokenização
        Retorno: uma lista com os tokens pré-processados
        '''
        # Retira todos os caracteres não alfabéticos
        values = self._regex['carac'].sub('', values.lower())
        # Retira os múltiplos espaços entre as palavras para deixar um espaço único
        values = self._regex['espaco'].sub(' ', values)
        # Gera lista de palavras excluídas as menores que min_len e maiores que max_len, sem acentuação
        words = g_utils.simple_preprocess(values, deacc=True, min_len=self.min_len, max_len=self.max_len)
        # Retorna as palavras pré-processadas tokenizadas com o seu atributo
        return [f'{atributo}_{word}' for word in words]

    def _povoar_atributos(self, values, zero=False):
        '''
        Povoa os atributos com os valores do dicionário passado. Esse é um método genérico para setar e persistir configurações.
        Parâmetros:
            values (dict) --> Dicionário onde a chave é o nome do atributo e o valor é o que se deseja armazenar
            zero (boolean) --> Indica se substitui None por zero (True) ou não (False) (default: False)
        Retorno: None
        '''
        for k, v in values.items():
            if zero and v is None: v = 0
            if isinstance(v, str): exec(f'self.{k} = "{v}"')
            else: exec(f'self.{k} = {v}')
        self._salvar_configuracoes()

    def _salvar_relacionamentos(self, ficha):
        '''
        Esse método é implementado em CorpusDimensao
        '''
        pass

    def _salvar_configuracoes(self):
        '''
        Persiste os dados de configuração do corpus.
        Retorno: None
        '''
        config = dict(nome = self.nome
                     ,projeto = self.projeto
                     ,acentos = self.acentos
                     ,tags_relac = self.tags_relac
                     ,min_len = self.min_len
                     ,max_len = self.max_len
                     ,no_below = self.no_below
                     ,no_above = self.no_above
                     ,keep_n = self.keep_n
                     ,num_docs = self.num_docs
                     ,num_atributos = self.num_atributos
                     ,num_fichas = self.num_fichas
                     ,num_words = self.num_words
                     ,num_tokens_full = self.num_tokens_full
                     ,freq_max_token_full = self.freq_max_token_full
                     ,freq_min_token_full = self.freq_min_token_full
                     ,num_fichas_token_full = self.num_fichas_token_full
                     ,max_tokens_ficha_full = self.max_tokens_ficha_full
                     ,min_tokens_ficha_full = self.min_tokens_ficha_full
                     ,avg_tokens_ficha_full = self.avg_tokens_ficha_full
                     ,sdv_tokens_ficha_full = self.sdv_tokens_ficha_full
                     ,num_tokens = self.num_tokens
                     ,freq_max_token = self.freq_max_token
                     ,freq_min_token = self.freq_min_token
                     ,num_fichas_token = self.num_fichas_token
                     ,max_tokens_ficha = self.max_tokens_ficha
                     ,min_tokens_ficha = self.min_tokens_ficha
                     ,avg_tokens_ficha = self.avg_tokens_ficha
                     ,sdv_tokens_ficha = self.sdv_tokens_ficha
                     ,_atributo_ficha = self._atributo_ficha
                     ,_lendo_csv = self._lendo_csv
                     ,_arquivo_csv = self._arquivo_csv
                     ,_sep = self._sep
                     ,_nrows = self._nrows
                     ,_total_docs = self._total_docs
                     ,_docs_lidos = self._docs_lidos
                     ,_id_origem = self._id_origem
                     ,_atributos = self._atributos
                     ,_has_dict = self._has_dict)
        with shelve.open(self._arqs['shelve']) as db:
            db[self._shelf] = config

    def _tratar_relacionamentos(self, values, atributo, new_atributo):
        '''
        Realiza o tratamento correspondente aos relacionamentos substituindo o atributo original pelo tipo de pessoa
        do relacionamento.
        Parâmetros:
            values (list de string) --> Tokens nos quais há uma pessoa informada
            atributo (string) --> Atributo original do token
            new_atributo (string) --> O tipo de pessoa que se encontra no relacionamento
        Retorno: a string original com novos tokens a serem incluídos no documento em Copus e None em CorpusDimensao
        '''
        return [value.replace(atributo, new_atributo) for value in values]


class CorpusDimensao(Corpus):
    '''
    Representa o corpus de uma dimensão, seu dicionário de palavras e suas respresentações vetorizadas. A abordagem do
    corpus separado em dimensões permite a consulta por semelhanças entre fichas que tem as mesmas dimensões.
    As informações de relacionamentos identificados dos dados dos documentos de todos os CPFs e CNPJs encontrados nos corpus
    (atributos que contém cpf ou cnpj no nome) são informadas no corpus da dimensão "Relacionamentos". A dimensão
    "Relacionamentos" é reservada e, se instanciada, não poderá ter documentos incluídos ao seu corpus, pois isso acontece a
    partir de outras dimensões.
    Os documentos são agrupados em fichas cujos nomes devem ser strings. A posição da ficha no corpus é a mesma da representação
    vetorizada dela, de modo a permitir a sua identificação nas rotinas de busca de fichas semelhantes.
    Parâmetros:
        projeto (string) --> Nome do projeto ao qual pertence o corpus. Não é sensitive case, ignora acentos e quantidade de
                espaços entre as palavras
        nome (string) --> Nome do corpus. Não é sensitive case, ignora acentos e quantidade de espaços entre as palavras
                (default: "Geral")
    Atributos:
        nome (string) --> Nome da dimensão
        projeto (string) --> Nome do projeto ao qual pertence o corpus
        acentos (Lista de string) --> Lista dos atributos, além dos "word", de cujos valores devem ser excluídos os acentos (default: lista vazia)
        tags_relac (list de string) --> Lista de palavras chaves que identificam relacionamentos em atributos
        min_len (int) --> Tamanho mínimo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word" (default: 4)
        max_len (int) --> Tamanho máximo de caracteres em uma palavra, considerando apenas palavras e números, para virar token de um atributo "word" (default: 100)
        no_below (int) --> Número mínimo de documentos no qual o token tem que aparecer para seguir no dicionário (default: 5)
        no_above (float) --> Percentual máximo de documentos no qual o token pode aparecer para seguir no diconário (default: 0.8)
        keep_n (int) --> Número máximo de tokens mais frequentes que seguirão no dicionário (default: 100000)
        num_tokens_full (int) --> Número de tokens no dicionário antes da filtragem
        freq_max_token_full (int) --> Número de documentos no qual aparece o token mais frequente antes da filtragem
        freq_min_token_full (int) --> Número de documentos no qual aparece o token menos frequente antes da filtragem
        num_tokens (int) --> Número de tokens após a filtragem
        freq_max_token (int) --> Número de documentos no qual aparece o token mais frequente após a filtragem
        freq_min_token (int) --> Número de documentos no qual aparece o token menos frequente após a filtragem
        num_docs (int) --> Total de documentos lidos para montar o corpus (linhas de CSVs)
        num_atributos (int) --> Total de atributos constantes do corpus
        num_fichas (int) --> Total de fichas do corpus
        num_words (int) --> Total de palavras processadas no corpus
    '''
    def __init__(self, projeto, nome):
        super().__init__(projeto, nome)
        # Instancia o CorpusDimensão "relacionamentos" se o corpus não for o de relacionamentos
        if self._link_nome == 'relacionamentos': self._dim_relac = None
        else: self._dim_relac = CorpusDimensao(nome='Relacionamentos', projeto=self.projeto)

    def _encerrar_incluir_documentos(self):
        '''
        Encerra o método incluir_documentos_csv.
        Retorno: None
        '''
        # Obtém os dados do corpus
        self._obter_dados_corpus()
        # Monta o dicionário
        self.montar_dicionario()
        # Cria as versões vetorizadas do corpus
        self.vetorizar()
        # Verifica se tem que atualizar os relacionamentos
        if self._dim_relac._lendo_csv:
            # Atualiza na dimensão relacionamentos o número de documentos lidos
            self._dim_relac.num_docs += self._docs_lidos
            self._dim_relac._lendo_csv = False
            self._dim_relac._salvar_configuracoes()
            # Obtém os dados do corpus da dimensão relacionamentos
            self._dim_relac._obter_dados_corpus()
            # Monta o dicionário da dimensão relacionamentos
            self._dim_relac.montar_dicionario()
            # Cria as versões vetorizadas do corpus
            self._dim_relac.vetorizar()

    def _salvar_relacionamentos(self, ficha):
        '''
        Executa o registro dos relacionamentos encontrados na dimensão relacionamentos
        Parâmetros:
            ficha (string) --> A ficha a qual pertencem as ocorrências que serão registradas
        Retorno: None
        '''
        # Anota que está sendo inserido dados em relacionamentos
        if not self._dim_relac._lendo_csv:
            self._dim_relac._lendo_csv = True
            self._dim_relac._id_origem = self._dim_relac._dao.registrar_origem(self._arquivo_csv, self._nrows)
            self._dim_relac._salvar_configuracoes()
        # Obtém o id da ficha para a dimensão relacionamentos
        id_ficha = self._dim_relac._dao.registrar_ficha(ficha)
        # Monta a lista de ocorrências a serem registradas
        ocorrencias = [(self._dim_relac._id_origem, id_ficha, self._dim_relac._tokens[token]['id_atributo'],
                        self._dim_relac._tokens[token]['id_token'], self._dim_relac._tokens[token]['freq_token'])
                        for token in self._dim_relac._tokens]
        # Regista na base do corpus as ocorrências
        self._dim_relac._dao.registrar_frequencias(ocorrencias)
        # Resseta o indicador de atualização da dimensão relacionamentos
        self._dim_relac._tokens = {}
        self._update_relac = False

    def _tratar_relacionamentos(self, values, atributo, new_atributo):
        '''
        Realiza o tratamento correspondente aos relacionamentos substituindo o atributo original pelo tipo de pessoa
        do relacionamento.
        Parâmetros:
            values (list de string) --> string com os tokens nos quais há uma pessoa informada
            atributo (string) --> Atributo original do token
            new_atributo (string) --> O tipo de pessoa que se encontra no relacionamento
        Retorno: a string original com novos tokens a serem incluídos no documento em Copus e None em CorpusDimensao
        '''
        if not self._update_relac: self._update_relac = True
        if not self._dim_relac._atributos.get(new_atributo):
            self._dim_relac._atributos[new_atributo] = self._dim_relac._dao.registrar_atributo(new_atributo)
        self._dim_relac._agregar_tokens([value.replace(atributo, new_atributo) for value in values],
                                        self._dim_relac._atributos[new_atributo])
