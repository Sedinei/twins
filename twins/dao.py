# Imports Python
import datetime as dt
import math
# Imports Twins
from twins.utils import ConexaoDB

class DAOCorpus:
    '''
    Interface entre a aplicação e o banco de dados onde estão persistidos alguns atributos do objeto Corpus ou CorpusDimensao.
    '''
    def __init__(self, arq_db):
        self._conn = ConexaoDB(arq_db)

    def consultar_db(self, sql, t=None):
        '''
        Método para realização de consultas SQL genéricas no DB.
        Parâmetros:
            sql (string) --> Código SQL a ser consultado com ou sem "?"
            t (tupla) --> Se o SQL tiver "?", são os valores que deverão ser substituídos no código
        Retorno: lista de tuplas, sendo que cada tupla representa uma linha do resultado da consulta
        '''
        with self._conn as c:
            if t: c.execute(sql, t)
            else: c.execute(sql)
            values = c.fetchall()
        return values

    def iniciar_dao(self):
        '''
        Verifica se a base existe, criando-a se não existe.
        Retorno:
            True --> A base existe e pode ser acessados os últimos dados
            False --> A base foi recém criada e deve ser povoada com os dados atuais, se for o caso.
        '''
        if self._conn.existe(): return True
        with self._conn as c:
            # Cria a tabela origens (ID criado automaticamente)
            c.execute('''CREATE TABLE origens (
                             id_origem INTEGER PRIMARY KEY
                            ,origem TEXT
                            ,nrows INTEGER
                            ,data TEXT)''')
            # Cria a tabela fichas (A APLICAÇÃO CONTROLA O ID)
            c.execute('''CREATE TABLE fichas (
                             id_ficha INTEGER PRIMARY KEY
                            ,ficha TEXT) WITHOUT ROWID''')
            # Cria a tabela atributos (ID criado automaticamente)
            c.execute('''CREATE TABLE atributos (
                             id_atributo INTEGER PRIMARY KEY
                            ,atributo TEXT)''')
            # Cria a tabela tokens (ID criado automaticamente)
            c.execute('''CREATE TABLE tokens (
                             id_token INTEGER PRIMARY KEY
                            ,token TEXT)''')
            # Cria a tabela corpus (base para o dicionário)
            c.execute('''CREATE TABLE corpus (
                             id_corpus INTEGER PRIMARY KEY
                            ,id_ficha INTEGER
                            ,id_token INTEGER
                            ,freq_token INTEGER)''')
            # Cria a tabela detalhes (estatísticas mais detalhadas)
            c.execute('''CREATE TABLE detalhes (
                             id_detalhe INTEGER PRIMARY KEY
                            ,id_origem INTEGER
                            ,id_ficha INTEGER
                            ,id_atributo INTEGER
                            ,id_token INTEGER
                            ,freq_token INTEGER)''')
            # Cria a tabela dos tokens do dicionário
            c.execute('''CREATE TABLE tokens_dict (
                             id_token INTEGER
                            ,token TEXT
                            ,num_fichas INTEGER
                            ,id_token_dict INTEGER)''')
            # Cria a tabela do dicionário do corpus
            c.execute('''CREATE TABLE bow_corpus (
                             id_ficha INTEGER
                            ,id_token_dict INTEGER
                            ,ficha TEXT
                            ,token TEXT
                            ,freq_token INTEGER)''')
            # Cria os índices para a tabela origens
            c.execute('CREATE INDEX origens_idx ON origens (origem)')
            # Cria os índices para a tabela fichas
            c.execute('CREATE INDEX fichas_idx ON fichas (ficha)')
            # Cria os índices para a tabela atributos
            c.execute('CREATE INDEX atributos_idx ON atributos (atributo)')
            # Cria os índices para a tabela tokens
            c.execute('CREATE INDEX tokens_idx ON tokens (token)')
            # Cria os índices para a tabela corpus
            c.execute('CREATE INDEX corpus_token_idx ON corpus (id_token)')
            c.execute('CREATE INDEX corpus_idx ON corpus (id_token, id_ficha)')
            # Cria os índices para a tabela tokens_dict
            c.execute('CREATE INDEX tokens_dict_id_idx ON tokens_dict (id_token)')
            c.execute('CREATE INDEX tokens_dict_token_idx ON tokens_dict (token)')
            # Cria os índices para a tabela dicionario
            c.execute('CREATE INDEX bow_corpus_ficha_idx ON bow_corpus (ficha)')
            c.execute('CREATE INDEX bow_corpus_id_ficha_idx ON bow_corpus (id_ficha)')
        return False

    def montar_dicionario(self, no_below, no_above, keep_n):
        '''
        Monta o dicionário do corpus e aplica os filtro dos parâmetros.
        Parâmetros:
            no_below (int) --> Número mínimo de documentos no qual o token tem que aparecer para seguir no dicionário
            no_above (float) --> Percentual máximo de documentos no qual o token pode aparecer para seguir no diconário
            keep_n (int) --> Número máximo de tokens mais frequentes que seguirão no dicionário
        Retorno: um dicionário com as estatísticas do dicionário de palavras antes e depois da filtragem
        '''
        est = {}
        with self._conn as c:
            # Obtém o número de fichas do corpus
            c.execute('SELECT count(*) FROM fichas')
            num_fichas = c.fetchone()[0]
            # Apaga o conteúdo das tabelas do dicionário
            c.execute('DELETE FROM tokens_dict')
            c.execute('DELETE FROM bow_corpus')
            # Obtém as estatísticas do dicionário sem filtragem
            c.execute('''WITH group_tokens_fichas AS (
                            SELECT id_token, count(id_ficha) AS num_fichas
                            FROM corpus GROUP BY 1)
                        SELECT count(id_token), min(num_fichas), max(num_fichas)
                        FROM group_tokens_fichas''')
            values = c.fetchone()
            est['num_tokens_full'] = values[0]
            est['freq_min_token_full'] = values[1]
            est['freq_max_token_full'] = values[2]
            # Obtém o número de fichas com ao menos um token
            c.execute('''WITH unique_fichas AS (
                            SELECT DISTINCT id_ficha
                            FROM corpus)
                        SELECT count(*) FROM unique_fichas''')
            est['num_fichas_token_full'] = c.fetchone()[0]
            # Obtém a maior e a menor quantidade de tokens em uma ficha
            c.execute('''WITH fichas_tokens AS (
                            SELECT id_ficha, count(id_token) AS qtd_tokens
                            FROM corpus GROUP BY 1)
                        SELECT min(qtd_tokens), max(qtd_tokens), avg(qtd_tokens)
                        FROM fichas_tokens''')
            values = c.fetchone()
            est['min_tokens_ficha_full'] = values[0]
            est['max_tokens_ficha_full'] = values[1]
            est['avg_tokens_ficha_full'] = values[2]
            # Obtém o desvio padrão da quantidade de tokens por ficha
            t = (est['avg_tokens_ficha_full'], est['avg_tokens_ficha_full'])
            c.execute('''WITH fichas_tokens AS (
                            SELECT id_ficha, count(id_token) AS qtd_tokens
                            FROM corpus GROUP BY 1)
                        SELECT avg((qtd_tokens - ?) * (qtd_tokens - ?))
                        FROM fichas_tokens''', t)
            value = c.fetchone()
            if not value[0]: est['sdv_tokens_ficha_full'] = None
            else: est['sdv_tokens_ficha_full'] = math.sqrt(value[0])
            # Constói a tabela tokens_dict aplicando os filtros
            max_fichas = int(num_fichas*no_above)
            t = (no_below, max_fichas, keep_n)
            c.execute('''INSERT INTO tokens_dict
                         SELECT tab1.id_token
                               ,tab2.token
                               ,count(tab1.id_ficha) AS num_fichas
                               ,null
                         FROM corpus AS tab1
                         LEFT JOIN tokens AS tab2
                            ON tab1.id_token=tab2.id_token
                         GROUP BY 1,2
                         HAVING num_fichas BETWEEN ? AND ?
                         LIMIT ?''', t)
            # Obtém as estatísticas do dicionário filtrado
            c.execute('SELECT count(id_token), min(num_fichas), max(num_fichas) FROM tokens_dict')
            values = c.fetchone()
            est['num_tokens'] = values[0]
            est['freq_min_token'] = values[1]
            est['freq_max_token'] = values[2]
            # Reindexa os tokens para os índices ficarem na dimensionalidade do dicionário
            c.execute('UPDATE tokens_dict SET id_token_dict=rowid-1')
            # Monta a tabela do dicionário para a montagem do corpus no formato BOW
            c.execute('''INSERT INTO bow_corpus
                         SELECT tab2.id_ficha
                               ,tab1.id_token_dict
                               ,tab3.ficha
                               ,tab1.token
                               ,tab2.freq_token
                         FROM tokens_dict AS tab1
                         LEFT JOIN corpus AS tab2
                            ON tab1.id_token=tab2.id_token
                         LEFT JOIN fichas AS tab3
                            ON tab2.id_ficha=tab3.id_ficha''')
            # Obtém o número de fichas com ao menos um token após a filtragem
            c.execute('''WITH unique_fichas AS (
                            SELECT DISTINCT id_ficha
                            FROM bow_corpus)
                        SELECT count(*) FROM unique_fichas''')
            est['num_fichas_token'] = c.fetchone()[0]
            # Obtém a maior e a menor quantidade de tokens em uma ficha após a filtragem
            c.execute('''WITH fichas_tokens AS (
                            SELECT id_ficha, count(id_token_dict) AS qtd_tokens
                            FROM bow_corpus GROUP BY 1)
                        SELECT min(qtd_tokens), max(qtd_tokens), avg(qtd_tokens)
                        FROM fichas_tokens''')
            values = c.fetchone()
            est['min_tokens_ficha'] = values[0]
            est['max_tokens_ficha'] = values[1]
            est['avg_tokens_ficha'] = values[2]
            # Obtém o desvio padrão da quantidade de tokens por ficha
            t = (est['avg_tokens_ficha_full'], est['avg_tokens_ficha_full'])
            c.execute('''WITH fichas_tokens AS (
                            SELECT id_ficha, count(id_token_dict) AS qtd_tokens
                            FROM bow_corpus GROUP BY 1)
                        SELECT avg((qtd_tokens - ?) * (qtd_tokens - ?))
                        FROM fichas_tokens''', t)
            value = c.fetchone()
            if not value[0]: est['sdv_tokens_ficha'] = None
            else: est['sdv_tokens_ficha'] = math.sqrt(value[0])
        return est

    def obter_atributos(self):
        '''
        Retorna a lista de atributos que existem no corpus.
        Retorno: lista de atributos (lista de string)
        '''
        with self._conn as c:
            c.execute('SELECT atributo FROM atributos')
            values = c.fetchall()
        return [att[0] for att in values]

    def obter_bow_ficha(self, ficha):
        '''
        Obtém o documento no formato BOW para a ficha informada. Se não encontrar a ficha no dicionário, retorna uma lista vazia.
        Retorno: uma lista de tuplas ou uma lista vazia
        '''
        with self._conn as c:
            t = (ficha, )
            c.execute('SELECT id_token_dict, freq_token FROM bow_corpus WHERE ficha=?', t)
            return c.fetchall()

    def obter_bow_id_ficha(self, id_ficha):
        '''
        Obtém o documento no formato BOW para o id_ficha informado. Se não encontrar o id_ficha no dicionário, retorna uma lista vazia.
        Retorno: uma lista de tuplas ou uma lista vazia
        '''
        with self._conn as c:
            t = (id_ficha, )
            c.execute('SELECT id_token_dict, freq_token FROM bow_corpus WHERE id_ficha=?', t)
            return c.fetchall()

    def obter_tokens_id_ficha(self, id_ficha):
        '''
        Retorna um dicionário com o token (string) como chave e a sua frequência (int) como valor para um dado id de ficha.
        Retorno: dicionário (string-->int) com os tokens e suas frequências para um certo id_ficha
        '''
        with self._conn as c:
            t = (id_ficha, )
            c.execute('SELECT token, freq_token FROM bow_corpus WHERE id_ficha=?', t)
            values = c.fetchall()
        return {token: freq_token for token, freq_token in values}
            
    def obter_bow_tokens(self, tokens):
        '''
        Obtém o documento no formato BOW para o string de tokens informados. Se nenhum dos tokens da string estiver no dicionário,
        retorna uma lista vazia.
        Retorno: uma lista de tuplas ou uma lista vazia
        '''
        # Monta um dicionário com os tokens e sua frequência
        tokens_freq = {}
        for token in tokens.split():
            tokens_freq[token] = tokens_freq.get(token, 0) + 1
        # Monta o BOW do documento consultando o id de cada token
        bow = []
        with self._conn as c:
            for token in tokens_freq:
                t = (token, )
                c.execute('SELECT id_token_dict FROM tokens_dict WHERE token=?', t)
                values = c.fetchone()
                if not values: continue
                bow.append((values[0], tokens_freq[token]))
        return bow

    def obter_dados_corpus(self):
        '''
        Calcula alguns dados referentes ao corpus.
        Retorno: dicionário com os dados referentes ao corpus
        '''
        est = {}
        with self._conn as c:
            # Obtém as quantidades para a estatística sendo a última o número de fichas 
            c.execute('SELECT count(*) FROM atributos')
            value = c.fetchone()[0]
            est['num_atributos'] = value
            c.execute('SELECT sum(freq_token) FROM corpus')
            value = c.fetchone()[0]
            est['num_words'] = value
            c.execute('SELECT count(*) FROM fichas')
            value = c.fetchone()[0]
            est['num_fichas'] = value
        return est

    def obter_fichas(self):
        '''
        Retorna a lista de fichas que constam do corpus na ordem em que elas aparecem do DB.
        Retorno: lista de fichas (lista de string)
        '''
        fichas = []
        with self._conn as c:
            c.execute('SELECT count(*) FROM fichas')
            num_fichas = c.fetchone()[0]
            for id_ficha in range(num_fichas):
                t = (id_ficha, )
                c.execute('SELECT ficha FROM fichas WHERE id_ficha=?', t)
                values = c.fetchone()
                if not values: fichas.append(None)
                else: fichas.append(values[0])
        return fichas

    def obter_ficha_id(self, id_ficha):
        '''
        Retorna o nome da ficha associada ao id passado no argumento.
        Parâmetros:
            id_ficha (int) --> Número do id da ficha que se deseja obter
        Retorno: o nome da ficha que corresponde ao id passado (string) ou None se não encontrar o id no corpus
        '''
        with self._conn as c:
            t = (id_ficha, )
            c.execute('SELECT ficha FROM fichas WHERE id_ficha=?', t)
            value = c.fetchone()
            if not value: return None
            else: return value[0]

    def obter_id_ficha(self, ficha):
        '''
        Retorna o índice do documento ao qual se refere a ficha passada em uma representação vetorial. Se a ficha não estiver presente
        no corpus, retorna None.
        Parâmetros:
            ficha (string) --> A ficha cujo id se deseja obter
        Retorno: o id da ficha passada (int)
        '''
        with self._conn as c:
            t = (ficha, )
            c.execute('SELECT id_ficha FROM fichas WHERE ficha=?', t)
            value = c.fetchone()
            if not value: return None
            else: return value[0]

    def obter_num_fichas_corpus(self):
        '''
        Retorna o número de fichas que há no corpus
        Retorno: número de fichas do corpus (int)
        '''
        with self._conn as c:
            c.execute('SELECT count(*) FROM fichas')
            return c.fetchone()[0]

    def obter_tokens_dicionario(self):
        '''
        Retorna um dicionário python a partir do dicionário do corpus onde a chave é o id do token e o valor é o token.
        Retorno: dicionário pyhton onde a chave é o id do token (int) e o valor é o token (string)
        '''
        with self._conn as c:
            c.execute('SELECT id_token_dict, token FROM tokens_dict GROUP BY 1,2')
            values = c.fetchall()
        return {id_token: token for id_token, token in values}

    def registrar_atributo(self, atributo):
        '''
        Registra o atributo, se for o caso, e retorna seu id.
        Parâmetros:
            atributo (String) --> Atributo cujo id se deseja obter
        Retorno: o id do atributo (Int)
        '''
        with self._conn as c:
            # Verifica se o atributo já está registrado
            t = (atributo, )
            sql_select = 'SELECT id_atributo FROM atributos WHERE atributo=?'
            c.execute(sql_select, t)
            num_id = c.fetchone()
            if num_id: num_id = num_id[0]
            else:
                # Registra o atributo e obtém o seu id
                sql_insert = 'INSERT INTO atributos VALUES (NULL,?)'
                c.execute(sql_insert, t)
                c.execute(sql_select, t)
                num_id = c.fetchone()[0]
        return num_id

    def registrar_ficha(self, ficha):
        '''
        Registra a ficha, se for o caso, e retorna seu id.
        Parâmetros:
            ficha (String) --> ficha cujo id se deseja obter
        Retorno: o id da ficha (Int)
        '''
        with self._conn as c:
            # Verifica se a ficha já está registrada
            t = (ficha, )
            c.execute('SELECT id_ficha FROM fichas WHERE ficha=?', t)
            num_id = c.fetchone()
            if num_id: num_id = num_id[0]
            else:
                # Obtém o próximo valor de índice e registra a nova ficha
                c.execute('SELECT count(*) FROM fichas')
                num_id = c.fetchone()[0]
                t = (num_id, ficha)
                c.execute('INSERT INTO fichas VALUES (?,?)', t)
        return num_id

    def registrar_frequencias(self, lst_values):
        '''
        Registra as frequencias encontradas na leitura dos tokens, acumulando os registros já existentes na base do corpus
        ou incluindo novos registros, se for o caso. Também povoa a base detalhes.
        Parâmetros:
            lst_values (Lista de tuplas) --> Cada tupla é formada do id_origem, id_ficha, id_atributo, id_token e freq_token
        Retorno: None
        '''
        with self._conn as c:
            for id_origem, id_ficha, id_atributo, id_token, freq_token in lst_values:
                # Verifica se o par id_ficha/id_token já está registrado no corpus
                t = (id_ficha, id_token)
                c.execute('SELECT id_corpus FROM corpus WHERE (id_ficha=? AND id_token=?)', t)
                num_id = c.fetchone()
                if not num_id:
                    # Registra a ocorrência em corpus
                    t = (id_ficha, id_token, freq_token)
                    c.execute('INSERT INTO corpus VALUES (NULL,?,?,?)', t)
                else:
                    # Obtém a frequencia anterior do token e a atualiza em corpus
                    num_id = num_id[0]
                    t = (num_id, )
                    c.execute('SELECT freq_token FROM corpus WHERE id_corpus=?', t)
                    t = (c.fetchone()[0] + freq_token, num_id)
                    c.execute('UPDATE corpus SET freq_token=? WHERE id_corpus=?', t)
                # Registra a ocorrência em detalhes
                t = (id_origem, id_ficha, id_atributo, id_token, freq_token)
                c.execute('INSERT INTO detalhes VALUES (NULL,?,?,?,?,?)', t)

    def registrar_lista_atributos(self, atributos):
        '''
        Registra os novos atributos em atributos, se for o caso, e retorna seus id.
        Parâmetros:
            atributos (Lista de String) --> Relação de atributos que se deseja obter os id
        Retorno: um dicionário com os ids (valores) dos atributos (chaves) (Dict)
        '''
        dic_ids = {}
        with self._conn as c:
            for value in atributos:
                # Verifica se o atributo/token já está registrado
                t = (value, )
                sql_select = 'SELECT id_atributo FROM atributos WHERE atributo=?'
                c.execute(sql_select, t)
                num_id = c.fetchone()
                if num_id: num_id = num_id[0]
                else:
                    # Registra o atributo e obtém seu id
                    sql_insert = 'INSERT INTO atributos VALUES (NULL,?)'
                    c.execute(sql_insert, t)
                    c.execute(sql_select, t)
                    num_id = c.fetchone()[0]
                dic_ids[value] = num_id
        return dic_ids

    def registrar_origem(self, origem, nrows):
        '''
        Registra a fonte de dados do corpus em origens.
        Parâmetros:
            origem (String) --> Endereço do arquivo do qual estão sendo obtidos os documentos para o corpus
        Retorno: o id da origem (Int)
        '''
        with self._conn as c:
            # Registra a origem
            data = str(dt.date.today())
            t = (origem, nrows, data)
            c.execute('INSERT INTO origens VALUES (null,?,?,?)', t)
            # Obtém o id da origem registada
            if not nrows:
                t = (origem, data)
                sql = 'SELECT id_origem FROM origens WHERE (origem=? AND data=? AND nrows is null)'
            else: sql = 'SELECT id_origem FROM origens WHERE (origem=? AND nrows=? AND data=?)'
            c.execute(sql, t)
            values = c.fetchall()
        num_id = values[len(values)-1][0]
        return num_id

    def registrar_token(self, token):
        '''
        Registra o token, se for o caso, e retorna seu id.
        Parâmetros:
            token (String) --> Token cujo id se deseja obter
        Retorno: o id do token (Int)
        '''
        with self._conn as c:
            # Verifica se o token já está registrado
            t = (token, )
            sql_select = 'SELECT id_token FROM tokens WHERE token=?'
            c.execute(sql_select, t)
            num_id = c.fetchone()
            if num_id: num_id = num_id[0]
            else:
                # Registra o token e obtém o seu id
                sql_insert = 'INSERT INTO tokens VALUES (NULL,?)'
                c.execute(sql_insert, t)
                c.execute(sql_select, t)
                num_id = c.fetchone()[0]
        return num_id
