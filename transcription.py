import whisper
import os
import torch

def transcrever_audio() -> str | None:
    """
    Pede ao usuário o nome de um arquivo de áudio/vídeo, transcreve usando Whisper
    e salva o resultado em 'data/transcricao.txt'.

    Retorna:
        O caminho para o arquivo de transcrição ('data/transcricao.txt') se bem-sucedido,
        None caso contrário.
    """
    arquivo_audio = input("Digite o nome do arquivo de áudio/vídeo (ex: audio.mp3, entrevista.wav, video.mp4): ").strip()

    caminho_completo = os.path.abspath(arquivo_audio)
    print(f"Verificando arquivo em: {caminho_completo}")

    if not os.path.isfile(caminho_completo):
        print(f"-> Arquivo não encontrado diretamente. Verificando na pasta 'data'...")
        caminho_na_data = os.path.abspath(os.path.join("data", arquivo_audio))
        print(f"Verificando arquivo em: {caminho_na_data}")
        if os.path.isfile(caminho_na_data):
            caminho_completo = caminho_na_data
            print("--> Arquivo encontrado na pasta 'data'.")
        else:
            print(f"Erro: Arquivo não encontrado em '{caminho_completo}' nem em '{caminho_na_data}'.")
            print("Verifique o nome e o local do arquivo.")
            return None # Retorna None em caso de falha

    pasta_saida = "data"
    os.makedirs(pasta_saida, exist_ok=True)
    arquivo_saida = os.path.join(pasta_saida, "transcricao.txt")

    print("\nCarregando modelo Whisper (isso pode levar alguns minutos na primeira vez)...")
    nome_modelo_whisper = "medium"

    if torch.cuda.is_available():
        device = "cuda"
        print("GPU (CUDA) detectada. Usando GPU para transcrição.")
    else:
        device = "cpu"
        print("GPU (CUDA) não detectada ou não suportada. Usando CPU para transcrição (pode ser lento).")

    try:
        model = whisper.load_model(nome_modelo_whisper, device=device)
        print(f"Modelo Whisper '{nome_modelo_whisper}' carregado no dispositivo '{device}'.")

    except Exception as e_load:
        print(f"Erro crítico ao carregar modelo Whisper: {e_load}")
        print("Verifique a instalação do Whisper, PyTorch e dependências (como ffmpeg).")
        import traceback
        traceback.print_exc()
        return None

    print(f"\nIniciando transcrição de '{os.path.basename(caminho_completo)}'...")
    print("Este processo pode demorar dependendo do tamanho do arquivo e do hardware.")
    try:
        use_fp16 = (device == "cuda")

        resultado = model.transcribe(caminho_completo, language='pt', fp16=use_fp16)

        with open(arquivo_saida, "w", encoding="utf-8") as f:
            f.write(resultado["text"])

        print(f"\nTranscrição concluída com sucesso!")
        print(f"Texto salvo em: {arquivo_saida}")
        return arquivo_saida

    except Exception as e_transcribe:
        print(f"\nErro durante a transcrição: {e_transcribe}")
        print("Verifique se o arquivo de áudio/vídeo é válido e se o `ffmpeg` está instalado e acessível no PATH do sistema.")
        import traceback
        traceback.print_exc()
        return None