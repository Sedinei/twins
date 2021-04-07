# Buscador de entidades similares - Twins

O Twins framework em estágio inicial de desenvolvimento que foi implementado como parte do Trabalho de Conclusão de Curso (TCC) de pós-graduação em Big Data da PUC-MG.
Trata-se de um framework para busca de entidades similares por meio da tokenização e vetorização de aspectos qualitativos das mesmas. As informações para cada entidade
podem ser fornecidas de forma incremental e essas informações podem todas serem agrupadas em um único corpus ou separadas em corpus que representam aspectos das entidades
que são chamados no framework de dimensões.
Os dados das entidades são colocados em fichas cujos nomes identificam cada entidade. Essas informações organizadas por atributos e são tokenizadas por meio da aglutinação
do nome do atributo com a informação em si. Podem ser indicados atributos para os quais serão removidos os acentos dos valores e atributos cujos valores provém de textos
livres e, portanto, precisam de um pré-processamento diferenciado (atributos "word").
É possível indicar tags que são palavras-chaves a serem buscadas nos atributos para identificar relacionamentos cuja semântica se quer reduzir para apenas a informação da
tag, o que amplia as possibilidades de consultas.
Foram implementados de maneira inicial a vetorização das informações nos formatos TF-IDF, TF-IDF com pivoteamento, LDA, LSI e Doc2Vec. Para isso foram utilizados os algoritmos
constantes do framework GenSim. O framework cria um embedding desses modelos (e das dimensões, se for adotada essa abordagem) para indicar a similaridade final de uma entidade
perante as demais que constam da base de dados.
