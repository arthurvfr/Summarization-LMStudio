import os
import requests
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, Generation
from typing import List
import re

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

def dividir_texto_robusto(texto: str, max_tokens: int = 500) -> List[str]:
    """
    Divide o texto em partes menores preservando:
    - Estrutura de parágrafos
    - Pontuação natural
    - Contexto semântico
    
    Args:
        texto: Texto completo a ser dividido
        max_tokens: Número máximo de tokens por segmento
        
    Returns:
        Lista de segmentos de texto bem divididos
    """
    # Pré-processamento: normaliza quebras de linha
    texto = re.sub(r'\r\n', '\n', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # Primeiro tenta dividir por parágrafos naturais
    segmentos = []
    paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
    
    current_segment = []
    current_length = 0
    
    for para in paragrafos:
        para_length = len(para.split())  # Estimativa simples de tokens
        
        if current_length + para_length <= max_tokens:
            current_segment.append(para)
            current_length += para_length
        else:
            if current_segment:
                segmentos.append('\n\n'.join(current_segment))
                current_segment = [para]
                current_length = para_length
            else:
                # Parágrafo muito grande, divide por sentenças
                sentencas = re.split(r'(?<=[.!?])\s+', para)
                sub_segment = []
                sub_length = 0
                
                for sent in sentencas:
                    sent_length = len(sent.split())
                    
                    if sub_length + sent_length <= max_tokens:
                        sub_segment.append(sent)
                        sub_length += sent_length
                    else:
                        if sub_segment:
                            segmentos.append(' '.join(sub_segment))
                            sub_segment = [sent]
                            sub_length = sent_length
                        else:
                            # Sentença muito grande, divide forçadamente
                            palavras = sent.split()
                            for i in range(0, len(palavras), max_tokens):
                                segmentos.append(' '.join(palavras[i:i+max_tokens]))
                
                if sub_segment:
                    segmentos.append(' '.join(sub_segment))
    
    if current_segment:
        segmentos.append('\n\n'.join(current_segment))
    
    return segmentos

def carregar_arquivo(nome_arquivo: str) -> str:
    """Versão melhorada com tratamento de encoding"""
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    try:
        with open(caminho, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(caminho, 'r', encoding='latin-1') as file:
            return file.read()

def salvar_arquivo(nome_arquivo: str, conteudo: str):
    """Salva conteúdo em um arquivo TXT na pasta data"""
    os.makedirs(DATA_DIR, exist_ok=True)
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    try:
        with open(caminho, 'w', encoding='utf-8') as file:
            file.write(conteudo)
    except OSError as e:
        if e.errno == 28:  # No space left on device
            raise Exception("Erro: Espaço em disco insuficiente") from e
        raise

def processar_entrevista_longa(texto: str, llm: LocalLLM) -> List[str]:
    """
    Processa entrevistas longas com divisão e processamento em lotes
    Retorna os trechos mais relevantes da entrevista
    """
    segmentos = dividir_texto_robusto(texto)
    trechos_relevantes = []
    
    for segmento in segmentos:
        # Processa cada segmento em paralelo (opcional)
        trechos = encontrar_trechos_relevantes([segmento], "", llm)
        trechos_relevantes.extend(trechos)
    
    return trechos_relevantes

def encontrar_trechos_relevantes(entrevistas: List[str], resumo_anterior: str, 
                               llm: LocalLLM, top_k: int = 3) -> List[str]:
    """Usa a própria LLm para encontrar trechos relevantes"""
    prompt = f"""
    [INSTRUÇÕES]
    1. Analise os trechos abaixo
    2. Identifique os {top_k} mais relevantes para complementar o resumo
    3. Mantenha o texto original sem alterações
    
    [RESUMO ANTERIOR]
    {resumo_anterior[:1000]}...  
    
    [ENTREVISTAS]
    {chr(10).join(entrevistas[:5])}  
    """
    
    resposta = llm._call(prompt)
    return processar_resposta_trechos(resposta, top_k)

def processar_resposta_trechos(resposta: str, top_k: int) -> List[str]:
    """Extrai trechos da resposta do LLM"""
    padrao = re.compile(r'\d+\.\s*(.+)$')
    trechos = []
    
    for linha in resposta.split('\n'):
        match = padrao.match(linha.strip())
        if match and len(trechos) < top_k:
            trechos.append(match.group(1))
    
    return trechos[:top_k]

def gerar_resumo_rag(entrevistas: List[str], resumo_anterior: str, llm: LocalLLM) -> str:
    """Gera o resumo implementando RAG"""
    
    trechos_relevantes = processar_entrevista_longa('\n\n'.join(entrevistas), llm)
    
    if len('\n'.join(trechos_relevantes).split()) > 2000:
        return gerar_resumo_em_partes(trechos_relevantes, resumo_anterior, llm)
    
    prompt = PromptTemplate(
        input_variables=["resumo", "contexto"],
        template="""
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
    )
    
    return llm._call(prompt.format(
        resumo=resumo_anterior[:3000],  
        contexto="\n---\n".join(trechos_relevantes)[:5000]  
    ))

def gerar_resumo_em_partes(trechos: List[str], resumo: str, llm: LocalLLM) -> str:
    """Gera resumo por partes para textos muito longos"""
    partes = dividir_texto_robusto('\n'.join(trechos), max_tokens=1500)
    resumo_atual = resumo
    
    for parte in partes:
        resumo_atual = gerar_resumo_rag([parte], resumo_atual, llm)
    
    return resumo_atual

if __name__ == "__main__":
    ARQUIVO_ENTREVISTA = "entrevista.txt"
    ARQUIVO_RESUMO_ANTERIOR = "resumo_anterior.txt"
    ARQUIVO_NOVO_RESUMO = "novo_resumo.txt"
    
    try:
        entrevista = carregar_arquivo(ARQUIVO_ENTREVISTA)
        resumo_anterior = carregar_arquivo(ARQUIVO_RESUMO_ANTERIOR)
        
        if len(entrevista.split()) > 5000:  
            print("Processando arquivo grande...")
            llm = LocalLLM(model_name="qwq-32b")
            novo_resumo = gerar_resumo_rag([entrevista], resumo_anterior, llm)
        else:
            entrevistas = [p.strip() for p in entrevista.split('\n\n') if p.strip()]
            llm = LocalLLM(model_name="qwq-32b")
            novo_resumo = gerar_resumo_rag(entrevistas, resumo_anterior, llm)
        
        salvar_arquivo(ARQUIVO_NOVO_RESUMO, novo_resumo)
        
        print(f"\n=== RESUMO GERADO COM SUCESSO ===")
        print(f"Arquivo criado em: {os.path.join(DATA_DIR, ARQUIVO_NOVO_RESUMO)}")
        print(f"Tamanho do novo resumo: {len(novo_resumo.split())} palavras")
        
    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo não encontrado na pasta 'data': {str(e)}")
        print(f"Certifique-se que os arquivos '{ARQUIVO_ENTREVISTA}' e '{ARQUIVO_RESUMO_ANTERIOR}' existem na pasta 'data'")
    except Exception as e:
        print(f"\nERRO durante o processamento: {str(e)}")