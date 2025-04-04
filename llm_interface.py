import requests
from langchain.llms.base import BaseLLM 
from langchain.schema import LLMResult, Generation
from typing import List, Optional, Dict, Any

from config import URL_LM_STUDIO

class LLMLocal(BaseLLM):
    identificador_modelo: str
    max_tokens_para_gerar: int = 2048
    temperatura: float = 0.3
    top_p: float = 0.9

    def __init__(self, config_modelo: Dict[str, Any], **kwargs):
        argumentos_init = {
            'identificador_modelo': config_modelo['nome'],
        }
        if 'max_tokens_output' in config_modelo:
             argumentos_init['max_tokens_para_gerar'] = config_modelo['max_tokens_output']
        if 'temperatura' in config_modelo:
            argumentos_init['temperatura'] = config_modelo['temperatura']
        if 'top_p' in config_modelo:
             argumentos_init['top_p'] = config_modelo['top_p']

        argumentos_init.update(kwargs)
        super().__init__(**argumentos_init)


    def _call(self, prompt: str) -> str:
        cabecalhos = {"Content-Type": "application/json"}
        dados_envio = {
            "model": self.identificador_modelo,
            "prompt": prompt,
            "max_tokens": self.max_tokens_para_gerar,
            "temperature": self.temperatura,
            "top_p": self.top_p,
        }

        try:
            resposta = requests.post(URL_LM_STUDIO, headers=cabecalhos, json=dados_envio, timeout=300)
            resposta.raise_for_status()

            resposta_json = resposta.json()
            escolhas = resposta_json.get('choices')
            if escolhas and isinstance(escolhas, list) and len(escolhas) > 0:
                texto_resposta = escolhas[0].get('text')
                if texto_resposta:
                    return texto_resposta.strip()

            print(f"Aviso: Formato inesperado da resposta do LLM: {resposta_json}")
            return ""

        except requests.Timeout:
             raise TimeoutError(f"Timeout: Requisição para o LM Studio excedeu 300 segundos.")
        except requests.RequestException as e:
            mensagem_erro = f"Erro ao conectar ao LM Studio em {URL_LM_STUDIO}. Verifique se está rodando e o modelo '{self.identificador_modelo}' está carregado.\nDetalhes: {e}"
            if resposta_erro := getattr(e, 'response', None):
                try:
                    detalhe_erro = resposta_erro.json()
                    mensagem_erro += f"\nResposta do LM Studio: {detalhe_erro}"
                except requests.exceptions.JSONDecodeError:
                    mensagem_erro += f"\nResposta do LM Studio (não-JSON): {resposta_erro.text}"
            raise ConnectionError(mensagem_erro) from e

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        resultados = []
        for prompt_atual in prompts:
            try:
                texto_gerado = self._call(prompt_atual, stop=stop)
                resultados.append([Generation(text=texto_gerado)])
            except Exception as e:
                print(f"Erro ao processar prompt: {prompt_atual[:100]}... \nErro: {e}")
                resultados.append([Generation(text=f"Erro ao processar: {e}")])
        # TODO: Adicionar token usage se a API retornar
        return LLMResult(generations=resultados)


    @property
    def _llm_type(self) -> str: 
        return "llm_local_lm_studio"