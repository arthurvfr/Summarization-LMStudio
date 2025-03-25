import requests
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, Generation

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
            "max_tokens": 1000,  # Aumentado para lidar com respostas maiores
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

# Entrevista longa para teste (aproximadamente 1000 tokens)
ENTREVISTA_LONGA = [
    """
    ENTREVISTA COM A CEO MARIA SILVA - 15/05/2024
    
    Durante nosso evento anual de inovação, a CEO Maria Silva compartilhou insights valiosos sobre os planos estratégicos:
    "Estamos investindo pesadamente em inteligência artificial, com um orçamento de R$ 50 milhões para os próximos 3 anos. 
    Nossa meta é integrar IA generativa em todos os produtos até o final de 2025. 
    Já temos 20 pesquisadores trabalhando no projeto Athena, nosso núcleo de inovação tecnológica."
    
    Sobre sustentabilidade: "Estamos comprometidos em reduzir nossa pegada de carbono em 40% até 2026. 
    Isso inclui a transição completa para energia renovável em todas as nossas instalações globais."
    
    Quando questionada sobre expansão: "América Latina é nossa prioridade imediata, com 3 novos escritórios 
    no Chile, Colômbia e México até Q2 do próximo ano. Também avaliamos oportunidades na Ásia-Pacífico."
    """,
    
    """
    RELATÓRIO FINANCEIRO - DIRETOR CARLOS MENDES
    
    O diretor financeiro apresentou números impressionantes:
    - Crescimento de 28% na receita no último trimestre
    - Margem de lucro aumentou para 18.7%
    - Investimentos em P&D atingiram 12% da receita
    
    "Nossas novas estratégias de precificação dinâmica e eficiência operacional estão gerando resultados. 
    Reduzimos custos operacionais em 15% enquanto aumentamos a produtividade em 22%", explicou Mendes.
    
    Sobre fusões e aquisições: "Estamos em negociações avançadas com duas startups de tecnologia 
    que complementariam nosso portfólio de produtos."
    """,
    
    """
    PESQUISA DE SATISFAÇÃO DOS FUNCIONÁRIOS
    
    A mais recente pesquisa interna revelou:
    - 92% de satisfação geral (aumento de 15% em relação a 2023)
    - 88% aprovaram as novas políticas de trabalho flexível
    - 85% sentem que a empresa investe no desenvolvimento profissional
    
    O diretor de RH comentou: "Nossos programas de mentoria e capacitação estão fazendo diferença. 
    A taxa de retenção de talentos subiu para 94%, a mais alta em 10 anos."
    
    Novas iniciativas anunciadas:
    - Programa de bolsas para mestrado e doutorado
    - Creche corporativa
    - Horário de trabalho personalizado
    """
]

# Resumo anterior para teste
RESUMO_ANTERIOR = """
A empresa está passando por um período de crescimento acelerado, com foco em inovação tecnológica.
Os indicadores financeiros mostram desempenho positivo, e a satisfação dos funcionários atingiu níveis recordes.
"""

if __name__ == "__main__":
    llm = LocalLLM(model_name="meta-llama-3.1-8b-instruct")
    
    try:
        print("=== PROCESSANDO ENTREVISTA LONGA ===")
        novo_resumo = gerar_resumo_rag(ENTREVISTA_LONGA, RESUMO_ANTERIOR, llm)
        
        print("\n=== RESUMO ATUALIZADO ===")
        print(novo_resumo)
        
        print(f"\nTamanho aproximado do resumo: {len(novo_resumo.split())} palavras")
    except Exception as e:
        print(f"Erro ao gerar resumo: {str(e)}")