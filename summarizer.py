import time
from typing import List, Optional
from langchain.prompts import PromptTemplate

from llm_interface import LLMLocal
from text_utils import dividir_texto_em_partes
from config import TAMANHO_PARTE_PALAVRAS_PADRAO, SOBREPOSICAO_PARTE_PALAVRAS_PADRAO

def resumir_parte(parte_texto: str, llm: LLMLocal, max_comprimento_resumo: int = 300) -> str:
    """Gera um resumo conciso para um único trecho de texto usando o LLM."""
    template_prompt = PromptTemplate(
        input_variables=["texto_parte", "max_comp"],
        template="""
        Leia o seguinte trecho de texto e gere um resumo conciso e informativo,
        capturando os pontos principais e informações chave.
        O resumo deve ter NO MÁXIMO {max_comp} palavras.
        Não adicione introduções como "Este é um resumo de..." ou "O trecho discute...".
        Apenas o resumo.

        TRECHO:
        {texto_parte}

        RESUMO CONCISO:
        """
    )
    prompt_formatado = template_prompt.format(texto_parte=parte_texto, max_comp=max_comprimento_resumo)
    resumo_gerado = llm._call(prompt_formatado)
    return resumo_gerado

def refinar_resumo(resumo_existente: str, resumo_nova_info: str, llm: LLMLocal, max_comprimento_final: Optional[int] = None) -> str:
    """Combina um resumo existente com novas informações, refinando o resultado."""
    if not resumo_nova_info or resumo_nova_info.strip().lower() in resumo_existente.lower():
        print("  (Refinamento pulado: Nova informação vazia ou já contida)")
        return resumo_existente

    instrucao_max_comp = ""
    if max_comprimento_final:
        instrucao_max_comp = f"O resumo final combinado NÃO DEVE EXCEDER {max_comprimento_final} palavras."

    template_prompt = PromptTemplate(
        input_variables=["resumo_atual", "resumo_novo", "instr_max_comp"],
        template="""
        Você está refinando um resumo existente. Combine o 'RESUMO ATUAL' com as 'NOVAS INFORMAÇÕES CHAVE' abaixo.
        O objetivo é criar um único resumo coeso e abrangente que integre as novas informações de forma natural,
        eliminando redundâncias e mantendo a clareza. Mantenha o tom e estilo do resumo atual.
        {instr_max_comp}
        Não inclua frases como "Este resumo combinado..." ou "Adicionando as novas informações...".
        Apresente APENAS o resumo final atualizado.

        RESUMO ATUAL:
        {resumo_atual}

        NOVAS INFORMAÇÕES CHAVE:
        {resumo_novo}

        RESUMO FINAL COMBINADO E REFINADO:
        """
    )

    max_palavras_input_estimado = llm.max_tokens_para_gerar * 2 
    palavras_prompt_base = len(template_prompt.template.split()) 
    palavras_resumo_novo = len(resumo_nova_info.split())

    palavras_restantes_para_existente = max_palavras_input_estimado - palavras_prompt_base - palavras_resumo_novo

    resumo_existente_para_prompt = resumo_existente
    if palavras_restantes_para_existente <= 0:
         print("  Aviso: Sem espaço estimado para resumo existente no prompt de refinamento. Usando apenas nova info.")
         resumo_existente_para_prompt = "[Resumo anterior omitido devido ao limite de contexto estimado]"
    elif len(resumo_existente.split()) > palavras_restantes_para_existente:
        print(f"  Aviso: Truncando resumo existente no prompt de refinamento (>{palavras_restantes_para_existente} palavras permitidas).")
        resumo_existente_para_prompt = " ".join(resumo_existente.split()[:palavras_restantes_para_existente]) + "..."


    prompt_formatado = template_prompt.format(
        resumo_atual=resumo_existente_para_prompt,
        resumo_novo=resumo_nova_info,
        instr_max_comp=instrucao_max_comp
    )

    resumo_refinado = llm._call(prompt_formatado)
    return resumo_refinado


def processar_entrevista_em_lotes(
    texto_completo: str,
    resumo_anterior: str,
    llm: LLMLocal,
    tamanho_parte: int = TAMANHO_PARTE_PALAVRAS_PADRAO,
    sobreposicao_parte: int = SOBREPOSICAO_PARTE_PALAVRAS_PADRAO
) -> str:
    """
    Processa um texto longo em lotes (map-reduce) para gerar um resumo.
    Fase 1 (Map): Divide o texto e resume cada parte.
    Fase 2 (Reduce): Combina e refina os resumos intermediários sequencialmente.
    """
    print(f"Dividindo o texto em partes (aprox. {tamanho_parte} palavras, {sobreposicao_parte} sobreposição)...")
    partes_texto = dividir_texto_em_partes(texto_completo, tamanho_parte, sobreposicao_parte)
    num_partes = len(partes_texto)
    print(f"Texto dividido em {num_partes} partes.")

    if not partes_texto:
        print("Texto vazio ou não pôde ser dividido em partes.")
        return resumo_anterior

    resumos_mapeados = []
    print("\n--- Fase 1: Gerando resumos iniciais (Map) ---")
    tempo_inicio_map = time.time()
    for i, parte_atual in enumerate(partes_texto):
        print(f"  Processando parte {i + 1}/{num_partes}...")
        try:
            max_palavras_resumo_intermediario = max(150, tamanho_parte // 3)
            resumo_parte_atual = resumir_parte(parte_atual, llm, max_comprimento_resumo=max_palavras_resumo_intermediario)
            if resumo_parte_atual and resumo_parte_atual.strip():
                 resumos_mapeados.append(resumo_parte_atual)
                 print(f"    Resumo intermediário gerado ({len(resumo_parte_atual.split())} palavras).")
            else:
                 print("    Aviso: Resumo intermediário vazio recebido.")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERRO ao sumarizar parte {i + 1}: {e}")

    duracao_mapeamento = time.time() - tempo_inicio_map
    print(f"--- Fase 1 concluída em {duracao_mapeamento:.2f} segundos. {len(resumos_mapeados)} resumos intermediários gerados. ---")

    if not resumos_mapeados:
        print("Nenhum resumo intermediário foi gerado.")
        return resumo_anterior 

    print("\n--- Fase 2: Combinando e refinando resumos (Reduce) ---")
    tempo_inicio_reduce = time.time()

    resumo_atual_combinado = resumo_anterior 

    total_refinamentos = len(resumos_mapeados)
    for i, resumo_novo_parte in enumerate(resumos_mapeados):
        print(f"  Refinando resumo: {i + 1}/{total_refinamentos}...")
        try:
            resumo_atual_combinado = refinar_resumo(resumo_atual_combinado, resumo_novo_parte, llm)
            print(f"    Novo tamanho do resumo combinado: {len(resumo_atual_combinado.split())} palavras.")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ERRO ao refinar com o resumo {i + 1}: {e}")

    duracao_reducao = time.time() - tempo_inicio_reduce
    print(f"--- Fase 2 concluída em {duracao_reducao:.2f} segundos. ---")

    return resumo_atual_combinado