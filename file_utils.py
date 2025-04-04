import os
from config import DIRETORIO_DADOS 

def carregar_arquivo(nome_arquivo: str) -> str:
    """
    Carrega o conteúdo de um arquivo de texto localizado no DIRETORIO_DADOS.
    Tenta várias codificações comuns.
    """
    caminho_arquivo = os.path.join(DIRETORIO_DADOS, nome_arquivo)
    codificacoes_tentar = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for cod in codificacoes_tentar:
        try:
            with open(caminho_arquivo, 'r', encoding=cod) as arquivo:
                return arquivo.read()
        except FileNotFoundError:
             raise FileNotFoundError(f"ERRO: Arquivo não encontrado em '{caminho_arquivo}'")
        except UnicodeDecodeError:
            continue 
        except Exception as e:
             raise IOError(f"Erro ao ler o arquivo '{caminho_arquivo}' com codificação {cod}: {e}") from e

    raise UnicodeDecodeError(f"Não foi possível decodificar o arquivo '{caminho_arquivo}' com as codificações testadas: {codificacoes_tentar}")

def salvar_arquivo(nome_arquivo: str, conteudo: str):
    """
    Salva o conteúdo de texto em um arquivo no DIRETORIO_DADOS.
    Usa codificação UTF-8.
    """
    caminho_arquivo = os.path.join(DIRETORIO_DADOS, nome_arquivo)
    try:
        os.makedirs(DIRETORIO_DADOS, exist_ok=True)
        with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
            arquivo.write(conteudo)
        print(f"Arquivo salvo com sucesso em: {caminho_arquivo}")
    except OSError as e:
        if e.errno == 28: 
            raise OSError("Erro: Espaço em disco insuficiente para salvar o arquivo.") from e
        else:
            raise OSError(f"Erro ao salvar o arquivo '{caminho_arquivo}': {e}") from e
    except Exception as e:
         raise Exception(f"Erro inesperado ao salvar o arquivo '{caminho_arquivo}': {e}") from e