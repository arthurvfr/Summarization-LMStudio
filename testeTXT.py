import os
import requests
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, Generation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/completions"

class LocalLLM(BaseLLM):
    model_name: str

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name

    def _call(self, prompt: str, stop=None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9
        }
        try:
            response = requests.post(LM_STUDIO_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('text', '').strip()
        except requests.RequestException as e:
            raise Exception(f"Erro ao conectar ao LM Studio: {e}")

    def _generate(self, prompts, stop=None):
        results = []
        for prompt in prompts:
            text = self._call(prompt)
            results.append([Generation(text=text)])
        return LLMResult(generations=results)

    @property
    def _llm_type(self):
        return "local_llm"

def carregar_arquivo(nome_arquivo):
    """Carrega o conteúdo de um arquivo TXT da pasta data"""
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    with open(caminho, 'r', encoding='utf-8') as file:
        return file.read()

def salvar_arquivo(nome_arquivo, conteudo):
    """Salva conteúdo em um arquivo TXT na pasta data"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    with open(caminho, 'w', encoding='utf-8') as file:
        file.write(conteudo)

def encontrar_trechos_relevantes(entrevistas, resumo_anterior, llm, top_k=3):
    """Usa o próprio LLM para encontrar trechos relevantes"""
    prompt = f"""
    Dado este resumo: {resumo_anterior}
    
    Analise estas entrevistas e retorne APENAS os {top_k} trechos mais relevantes 
    (mantenha o texto original sem modificações):
    
    ENTREVISTAS:
    {chr(10).join(entrevistas)}
    
    Formato de resposta EXATO:
    1. [trecho mais relevante exatamente como no original]
    2. [segundo trecho mais relevante]
    3. [terceiro trecho mais relevante]
    """
    
    resposta = llm._call(prompt)
    return [linha.split('. ')[1] for linha in resposta.split('\n') if '. ' in linha][:top_k]

def gerar_resumo_rag(entrevistas, resumo_anterior, llm):
    """Implementação completa do RAG com modelo único"""
    # Passo 1: Recuperação
    trechos_relevantes = encontrar_trechos_relevantes(entrevistas, resumo_anterior, llm)
    
    # Passo 2: Geração
    prompt_template = """
    Com base no resumo atual e nos trechos relevantes das entrevistas, 
    produza um NOVO resumo detalhado (400-500 palavras) que:
    
    1. Mantenha todas informações importantes do resumo original
    2. Incorpore os novos insights dos trechos relevantes
    3. Seja bem estruturado e coeso
    4. Apresente apenas o resumo gerado, sem tópicos ou perguntas
    
    RESUMO ATUAL:
    {resumo}
    
    TRECHOS RELEVANTES:
    {contexto}
    
    NOVO RESUMO DETALHADO:
    """
    
    prompt = PromptTemplate(
        input_variables=["resumo", "contexto"],
        template=prompt_template
    )
    
    return llm._call(prompt.format(
        resumo=resumo_anterior,
        contexto="\n---\n".join(trechos_relevantes)
    ))

if __name__ == "__main__":
    ARQUIVO_ENTREVISTA = "entrevista.txt"
    ARQUIVO_RESUMO_ANTERIOR = "resumo_anterior.txt"
    ARQUIVO_NOVO_RESUMO = "novo_resumo.txt"
    
    try:
        # Carrega os arquivos da pasta data
        entrevista = carregar_arquivo(ARQUIVO_ENTREVISTA)
        resumo_anterior = carregar_arquivo(ARQUIVO_RESUMO_ANTERIOR)
        
        # Divide a entrevista em parágrafos
        entrevistas = [p.strip() for p in entrevista.split('\n\n') if p.strip()]
        
        llm = LocalLLM(model_name="qwq-32b")
        
        print("=== PROCESSANDO ARQUIVOS ===")
        print(f"Lendo entrevista de: {os.path.join(DATA_DIR, ARQUIVO_ENTREVISTA)}")
        novo_resumo = gerar_resumo_rag(entrevistas, resumo_anterior, llm)
        
        # Salva o novo resumo na pasta data
        salvar_arquivo(ARQUIVO_NOVO_RESUMO, novo_resumo)
        
        print(f"\n=== RESUMO GERADO COM SUCESSO ===")
        print(f"Arquivo criado em: {os.path.join(DATA_DIR, ARQUIVO_NOVO_RESUMO)}")
        print(f"Tamanho do novo resumo: {len(novo_resumo.split())} palavras")
        
    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo não encontrado na pasta 'data': {str(e)}")
        print(f"Certifique-se que os arquivos '{ARQUIVO_ENTREVISTA}' e '{ARQUIVO_RESUMO_ANTERIOR}' existem na pasta 'data'")
    except Exception as e:
        print(f"\nERRO durante o processamento: {str(e)}")