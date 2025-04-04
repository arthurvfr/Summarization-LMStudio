import os
import time
import traceback
from typing import Dict, Any

from config import (
    MODELOS_DISPONIVEIS,
    TAMANHO_PARTE_PALAVRAS_PADRAO,
    SOBREPOSICAO_PARTE_PALAVRAS_PADRAO,
    ARQUIVO_ENTREVISTA,
    ARQUIVO_RESUMO_ANTERIOR,
    DIRETORIO_DADOS,
    URL_LM_STUDIO
)


from llm_interface import LLMLocal
from file_utils import carregar_arquivo, salvar_arquivo
from summarizer import processar_entrevista_em_lotes
from transcription import transcrever_audio


def selecionar_modelo() -> Dict[str, Any]:
    print("\n--- Seleção de Modelo LLM para Resumo ---")
    for chave, modelo in MODELOS_DISPONIVEIS.items():
        print(f"{chave}. {modelo['nome']} (Saída: {modelo['max_tokens_output']}, Contexto ~{modelo['context_window_approx']})")

    while True:
        escolha_usuario = input(f"\nEscolha o modelo LLM ({', '.join(MODELOS_DISPONIVEIS.keys())}): ").strip()
        if escolha_usuario in MODELOS_DISPONIVEIS:
            modelo_selecionado = MODELOS_DISPONIVEIS[escolha_usuario]
            print(f"Modelo LLM selecionado: {modelo_selecionado['nome']}")

            palavras_aprox_prompt_map = TAMANHO_PARTE_PALAVRAS_PADRAO + 150
            if palavras_aprox_prompt_map > modelo_selecionado['context_window_approx']:
                 print("\n--- AVISO (Contexto LLM) ---")
                 print(f"O tamanho da parte ({TAMANHO_PARTE_PALAVRAS_PADRAO} palavras) + prompt pode exceder")
                 print(f"a janela de contexto estimada ({modelo_selecionado['context_window_approx']}) deste modelo LLM na fase de resumo inicial.")
                 print("Isso pode causar erros ou resultados truncados no resumo.")
                 print("Considere reduzir 'TAMANHO_PARTE_PALAVRAS_PADRAO' em config.py se ocorrerem problemas.")
                 print("---------------------------")

            return modelo_selecionado
        else:
            print(f"Opção inválida. Digite um número entre {min(MODELOS_DISPONIVEIS.keys())} e {max(MODELOS_DISPONIVEIS.keys())}.")


if __name__ == "__main__":
    try:
        print("\n==============================================")
        print(" Assistente de Transcrição e Resumo")
        print("==============================================")
        print("\n--- Opção Inicial ---")
        print("1. Transcrever um novo arquivo de áudio/vídeo e depois resumir.")
        print(f"2. Usar o arquivo de transcrição existente ('{ARQUIVO_ENTREVISTA}' na pasta 'data').")

        arquivo_transcricao_local = os.path.join(DIRETORIO_DADOS, ARQUIVO_ENTREVISTA)
        arquivo_transcricao_pronto = False

        while not arquivo_transcricao_pronto:
            escolha_acao = input("Digite sua escolha (1 ou 2): ").strip()

            if escolha_acao == '1':
                print("\n--- Iniciando Módulo de Transcrição ---")
                caminho_transcrito = transcrever_audio()

                if caminho_transcrito and os.path.exists(caminho_transcrito):
                    print("\n--- Transcrição bem-sucedida. Prosseguindo para o resumo. ---")
                    if os.path.abspath(caminho_transcrito) == os.path.abspath(arquivo_transcricao_local):
                        arquivo_transcricao_pronto = True
                    else:
                        print(f"Aviso: Caminho transcrito ({caminho_transcrito}) difere do esperado ({arquivo_transcricao_local}). Verifique o código.")
                        arquivo_transcricao_pronto = True
                else:
                    print("\nFalha na transcrição. Verifique os erros detalhados acima.")
                    print("Não é possível continuar sem um arquivo de transcrição válido.")
                    exit(1)


            elif escolha_acao == '2':
                print(f"\n--- Verificando arquivo existente: {arquivo_transcricao_local} ---")
                if not os.path.exists(arquivo_transcricao_local):
                     print(f"\nErro: O arquivo de transcrição '{ARQUIVO_ENTREVISTA}' não foi encontrado em '{DIRETORIO_DADOS}'.")
                     print("-> Certifique-se de que o arquivo existe ou escolha a opção 1 para transcrever primeiro.")
                elif os.path.getsize(arquivo_transcricao_local) == 0:
                     print(f"\nAviso: O arquivo de transcrição '{arquivo_transcricao_local}' existe, mas está vazio.")
                     print("-> Resumir um arquivo vazio não produzirá resultados úteis.")
                     print("-> Deseja continuar mesmo assim ou escolher outra opção?")
                     confirmar = input("Continuar com arquivo vazio? (s/N): ").strip().lower()
                     if confirmar == 's':
                         print("Continuando com arquivo vazio...")
                         arquivo_transcricao_pronto = True
                else:
                     print("--> Arquivo existente encontrado e não está vazio. Prosseguindo para o resumo.")
                     arquivo_transcricao_pronto = True

            else:
                print("Escolha inválida. Por favor, digite 1 ou 2.")

            if arquivo_transcricao_pronto:
                break


        config_modelo_escolhido = selecionar_modelo()
        nome_arquivo_saida = config_modelo_escolhido['nome_arquivo']

        print(f"\nCarregando arquivo de transcrição: {arquivo_transcricao_local}...")
        try:
            entrevista_completa = carregar_arquivo(ARQUIVO_ENTREVISTA)
            print(f"Tamanho da transcrição: {len(entrevista_completa)} caracteres, aprox. {len(entrevista_completa.split())} palavras.")
            if not entrevista_completa.strip() and escolha_acao != '2':
                 print("Aviso: O arquivo de transcrição carregado está vazio ou contém apenas espaços.")

        except FileNotFoundError:
             print(f"ERRO INESPERADO: Arquivo '{ARQUIVO_ENTREVISTA}' não encontrado em '{DIRETORIO_DADOS}' após seleção/transcrição.")
             exit(1)
        except Exception as e:
             print(f"Erro ao carregar o arquivo de transcrição '{ARQUIVO_ENTREVISTA}': {e}")
             exit(1)

        resumo_anterior = ""
        arquivo_resumo_ant_local = os.path.join(DIRETORIO_DADOS, ARQUIVO_RESUMO_ANTERIOR)
        try:
            print(f"Carregando resumo anterior (se existir): {arquivo_resumo_ant_local}...")
            resumo_anterior = carregar_arquivo(ARQUIVO_RESUMO_ANTERIOR)
            if resumo_anterior.strip():
                 print(f"Resumo anterior carregado ({len(resumo_anterior.split())} palavras).")
            else:
                 print("Arquivo de resumo anterior encontrado, mas está vazio.")
                 resumo_anterior = ""
        except FileNotFoundError:
            print("Arquivo de resumo anterior não encontrado. Iniciando com resumo vazio.")
        except Exception as e:
             print(f"Erro ao carregar resumo anterior ({ARQUIVO_RESUMO_ANTERIOR}): {e}. Iniciando com resumo vazio.")
             resumo_anterior = ""


        print("\nInicializando LLM para resumo...")
        llm = LLMLocal(config_modelo=config_modelo_escolhido)
        try:
             print("Testando conexão com LLM...")
             resposta_teste = llm._call("Responda APENAS com 'OK'.")
             if 'ok' in resposta_teste.strip().lower():
                 print(f"Conexão com LLM bem-sucedida. Resposta: '{resposta_teste}'")
             else:
                 print(f"Alerta: Conexão com LLM estabelecida, mas resposta inesperada ao teste: '{resposta_teste}'. Verifique o modelo e o prompt de teste.")

        except ConnectionError as erro_conexao:
             print(f"\n--- ERRO DE CONEXÃO LLM ---")
             print(f"Não foi possível conectar ao LLM do LM Studio ou obter resposta.")
             print(f"Verifique se o LM Studio está rodando, se o modelo '{config_modelo_escolhido['nome']}' está completamente carregado e se a URL '{URL_LM_STUDIO}' está correta.")
             print(f"Detalhes do erro: {erro_conexao}")
             print("----------------------------")
             exit(1)
        except Exception as erro_geral_teste:
            print(f"\n--- ERRO DURANTE TESTE DE CONEXÃO LLM ---")
            print(f"Ocorreu um erro ao tentar comunicar com o LLM para teste:")
            print(f"Erro: {erro_geral_teste}")
            print("-----------------------------------------")
            exit(1)


        print(f"\n=== INICIANDO RESUMO COM MODELO {llm.identificador_modelo} ===")
        tempo_inicio_processo = time.time()

        novo_resumo = processar_entrevista_em_lotes(
            texto_completo=entrevista_completa,
            resumo_anterior=resumo_anterior,
            llm=llm,
            tamanho_parte=TAMANHO_PARTE_PALAVRAS_PADRAO,
            sobreposicao_parte=SOBREPOSICAO_PARTE_PALAVRAS_PADRAO
        )

        duracao_processo = time.time() - tempo_inicio_processo
        print(f"\nProcessamento do resumo concluído em {duracao_processo:.2f} segundos.")

        arquivo_resumo_final_local = os.path.join(DIRETORIO_DADOS, nome_arquivo_saida)
        print(f"\nSalvando novo resumo em: {arquivo_resumo_final_local}...")
        salvar_arquivo(nome_arquivo_saida, novo_resumo)

        print(f"\n==============================================")
        print(f" OPERAÇÃO CONCLUÍDA COM SUCESSO")
        print(f"==============================================")
        print(f"Arquivo de resumo gerado/atualizado: {arquivo_resumo_final_local}")
        print(f"Tamanho do novo resumo: {len(novo_resumo.split())} palavras")

    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: Arquivo essencial não encontrado.")
        print(e)
    except ConnectionError as e:
         print(f"\nERRO CRÍTICO: Falha na comunicação com o LM Studio durante o processamento.")
         print(e)
    except OSError as e:
         print(f"\nERRO CRÍTICO: Problema de Sistema Operacional (ex: I/O, disco cheio?).")
         print(e)
    except Exception as e:
        print(f"\nERRO INESPERADO durante a execução:")
        print(traceback.format_exc())