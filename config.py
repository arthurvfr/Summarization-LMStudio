import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DIRETORIO_DADOS = os.path.join(BASE_DIR, 'data')

URL_LM_STUDIO = "http://127.0.0.1:1234/v1/completions"

MODELOS_DISPONIVEIS = {
    "1": {
        "nome": "gemma-3-12b-it@q8_0",
        "nome_arquivo": "resumo_gemma12B_8bits.txt",
        "max_tokens_output": 2048,
        "context_window_approx": 8000
    },
    "2": {
        "nome": "meta-llama-3.1-8b-instruct",
        "nome_arquivo": "resumo_llama8B.txt",
        "max_tokens_output": 2048,
        "context_window_approx": 8000
    },
    "3": {
        "nome": "gemma-3-27b-it",
        "nome_arquivo": "resumo_gemma27b.txt",
        "max_tokens_output": 4096,
        "context_window_approx": 8000
    },
    "4": {
        "nome": "deepseek-r1-distill-qwen-32b",
        "nome_arquivo": "resumo_qwen32b.txt",
        "max_tokens_output": 4096,
        "context_window_approx": 16000
    },
    "5": {
        "nome": "lmstudio-community/deepseek-r1-distill-qwen-32b",
        "nome_arquivo": "resumo_qwen32b_3bits.txt",
        "max_tokens_output": 4096,
        "context_window_approx": 16000
    }
}

TAMANHO_PARTE_PALAVRAS_PADRAO = 700
SOBREPOSICAO_PARTE_PALAVRAS_PADRAO = 100

ARQUIVO_ENTREVISTA = "entrevista.txt"
ARQUIVO_RESUMO_ANTERIOR = "resumo_anterior.txt"

os.makedirs(DIRETORIO_DADOS, exist_ok=True)