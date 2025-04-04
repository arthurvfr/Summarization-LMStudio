import re
from typing import List

from config import TAMANHO_PARTE_PALAVRAS_PADRAO, SOBREPOSICAO_PARTE_PALAVRAS_PADRAO

def limpar_texto(texto: str) -> str:
    """Limpa o texto removendo espaços extras e normalizando quebras de linha."""
    texto = re.sub(r'\r\n', '\n', texto) 
    texto = re.sub(r'\n{3,}', '\n\n', texto) 
    texto = re.sub(r' +', ' ', texto) 
    texto = texto.strip() 
    return texto

def dividir_texto_em_partes(
    texto: str,
    tamanho_parte_palavras: int = TAMANHO_PARTE_PALAVRAS_PADRAO,
    sobreposicao_parte_palavras: int = SOBREPOSICAO_PARTE_PALAVRAS_PADRAO
) -> List[str]:
    """
    Divide um texto longo em partes menores com sobreposição, tentando respeitar parágrafos
    e sentenças, com base em contagem de palavras.
    """
    if not texto:
        return []

    texto_limpo = limpar_texto(texto)
    paragrafos = [p for p in texto_limpo.split('\n\n') if p.strip()]
    if not paragrafos: 
        paragrafos = [texto_limpo]

    partes = []
    palavras_parte_atual = []
    contagem_palavras_atual = 0

    for paragrafo in paragrafos:
        palavras_paragrafo = paragrafo.split()
        if not palavras_paragrafo:
            continue 

        precisa_dividir_paragrafo = (contagem_palavras_atual > 0 and contagem_palavras_atual + len(palavras_paragrafo) > tamanho_parte_palavras) or \
                                     (len(palavras_paragrafo) > tamanho_parte_palavras * 1.2) # Adiciona uma folga para parágrafos um pouco maiores

        if precisa_dividir_paragrafo and palavras_parte_atual:
            partes.append(" ".join(palavras_parte_atual))
            indice_inicio_sobreposicao = max(0, len(palavras_parte_atual) - sobreposicao_parte_palavras)
            palavras_parte_atual = palavras_parte_atual[indice_inicio_sobreposicao:] 
            contagem_palavras_atual = len(palavras_parte_atual)

        if len(palavras_paragrafo) > tamanho_parte_palavras:
            sentencas = re.split(r'(?<=[.!?])\s+', paragrafo)
            parte_temporaria_sentenca = []
            contagem_palavras_temp_sentenca = 0

            for sentenca in sentencas:
                palavras_sentenca = sentenca.split()
                if not palavras_sentenca: continue

                if contagem_palavras_temp_sentenca + len(palavras_sentenca) <= tamanho_parte_palavras:
                    parte_temporaria_sentenca.extend(palavras_sentenca)
                    contagem_palavras_temp_sentenca += len(palavras_sentenca)
                else:
                    if parte_temporaria_sentenca:
                        palavras_combinadas = palavras_parte_atual + parte_temporaria_sentenca
                        partes.append(" ".join(palavras_combinadas))
                        indice_inicio_sobreposicao = max(0, len(palavras_combinadas) - sobreposicao_parte_palavras)
                        palavras_parte_atual = palavras_combinadas[indice_inicio_sobreposicao:]
                        contagem_palavras_atual = len(palavras_parte_atual)
                    else:
                        if palavras_parte_atual:
                            partes.append(" ".join(palavras_parte_atual))
                        palavras_parte_atual = []
                        contagem_palavras_atual = 0

                    for i in range(0, len(palavras_sentenca), tamanho_parte_palavras):
                        parte_palavras = palavras_sentenca[i:i+tamanho_parte_palavras]
                        palavras_combinadas_sentenca_split = palavras_parte_atual + parte_palavras
                        partes.append(" ".join(palavras_combinadas_sentenca_split))
                        indice_inicio_sobreposicao_sentenca = max(0, len(palavras_combinadas_sentenca_split) - sobreposicao_parte_palavras)
                        palavras_parte_atual = palavras_combinadas_sentenca_split[indice_inicio_sobreposicao_sentenca:]
                        contagem_palavras_atual = len(palavras_parte_atual)

                    parte_temporaria_sentenca = []
                    contagem_palavras_temp_sentenca = 0


            if parte_temporaria_sentenca:
                palavras_combinadas = palavras_parte_atual + parte_temporaria_sentenca
                partes.append(" ".join(palavras_combinadas))
                indice_inicio_sobreposicao = max(0, len(palavras_combinadas) - sobreposicao_parte_palavras)
                palavras_parte_atual = palavras_combinadas[indice_inicio_sobreposicao:]
                contagem_palavras_atual = len(palavras_parte_atual)

        else:
            palavras_parte_atual.extend(palavras_paragrafo)
            contagem_palavras_atual += len(palavras_paragrafo)

        if contagem_palavras_atual >= tamanho_parte_palavras:
            partes.append(" ".join(palavras_parte_atual))
            indice_inicio_sobreposicao = max(0, len(palavras_parte_atual) - sobreposicao_parte_palavras)
            palavras_parte_atual = palavras_parte_atual[indice_inicio_sobreposicao:]
            contagem_palavras_atual = len(palavras_parte_atual)

    if palavras_parte_atual:
        partes.append(" ".join(palavras_parte_atual))

    return partes