# Imports Python
import os
import json
import shelve
# Imports Twins
from twins.corpus import CorpusDimensao
from twins.utils import obter_link_name

class Dimensoes:
    '''
    Iterável de objetos CorpusDimensoes.
    '''
    def __init__(self, projeto):
        self._dimensoes = {}
        self._dados = {}
        self._projeto = projeto
        self._shelve = os.path.join(f'./projetos/{obter_link_name(self._projeto)}', 'objetos.db')
        # Iniciar dimensoes
        self._iniciar_dimensoes()

    def __contains__(self, nome):
        return obter_link_name(nome) in self._dados

    def __getitem__(self, nome):
        link_name = obter_link_name(nome)
        if link_name not in self._dados:
            print(f'A dimensão "{nome}" não foi incluída no controle.')
            return None
        return self._dimensoes[link_name]

    def __iter__(self):
        for dimensao in self._dimensoes:
            yield self._dimensoes[dimensao]

    def ajustar_pesos(self, **kwargs):
        '''
        Ajusta os pesos das dimensões conforme passado no dicionário.
        Parâmetros:
            O nome da dimensão e o peso que se quer atribuir a ela
        Retorno: None
        '''
        for nome, peso in kwargs.items():
            link_name = obter_link_name(nome)
            if link_name not in self._dados:
                print(f'A dimensão "{nome}" não foi incluída no controle.')
                continue
            self._dados[link_name]['peso'] = peso
        self._salvar_dimensoes()

    def incluir(self, nome):
        '''
        Inclui uma dimensão ao iterável.
        Parâmetros:
            nome (string) --> Nome da dimensão
        Retorno: None
        '''
        link_name = obter_link_name(nome)
        if link_name in self._dados:
            print(f'Essa dimensão já foi incluída sob o nome "{self._dados[link_name]["nome"]}".')
            return
        self._dimensoes[link_name] = CorpusDimensao(nome=nome, projeto=self._projeto)
        self._dados[link_name] = {'nome': nome, 'peso': 1.0}
        self._salvar_dimensoes()

    def peso(self, nome):
        '''
        Retorna o peso da dimensão.
        Parâmetros:
            nome (string) --> Nome da dimensão
        Retorno: o peso da dimensão (float) ou None se a dimensão não está incluída
        '''
        link_name = obter_link_name(nome)
        if link_name not in self._dados:
            print(f'Não foi incluida a dimensão de nome "{nome}".')
            return None
        return self._dados[link_name]['peso']

    def _iniciar_dimensoes(self):
        '''
        Verifica se já havia um objeto Dimensoes instanciado e, se sim, recupera as configurações anteriores.
        Retorno: None
        '''
        # Verifica se há configurações anteriores para serem recuperadas
        if not os.path.isfile(f'{self._shelve}.dat'): return
        # Verifica se há dados sobre dimensoes no arquivo shelve
        with shelve.open(self._shelve) as db:
            if 'dimensoes' not in db: return
            dados = db['dimensoes']
        # Recupera as configurações anteriores de dimensoes
        pesos = {}
        for dimensao in dados.values():
            self.incluir(dimensao['nome'])
            pesos[dimensao['nome']] = dimensao['peso']
        self.ajustar_pesos(**pesos)
        # Atualiza o arquivo de configurações com a versão atual da classe
        self._salvar_dimensoes()

    def _salvar_dimensoes(self):
        '''
        Persiste os dados do objeto para recuperação futura.
        Retorno: None
        '''
        with shelve.open(self._shelve) as db:
            db['dimensoes'] = self._dados
