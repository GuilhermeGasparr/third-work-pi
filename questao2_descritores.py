import json
import os

import numpy as np
from PIL import Image 
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(PASTA, "resultados")
os.makedirs(SAIDA, exist_ok=True)

IMAGENS = {
    "ia": ("Imagem 1 (IA, cinematográfica)", os.path.join(PASTA, "assets", "imagem_ia_cinematografica.jpg")),
    "ilustracao": ("Imagem 2 (ilustração digital)", os.path.join(PASTA, "assets", "imagem_ilustracao_digital.jpg")),
}

REGIOES = {
    "ia": [
        ("Fundo desfocado (muro)", (0, 300, 0, 450)),
        ("Rosto", (60, 420, 700, 1000)),
        ("Roupa e corrente", (560, 760, 850, 1300)),
    ],
    "ilustracao": [
        ("Fundo azul (céu)", (190, 280, 320, 640)),
        ("Grama pintada", (330, 520, 330, 540)),
        ("Casal e banco", (270, 640, 520, 1060)),
        ("Reflexo borrado", (620, 768, 300, 1050)),
    ],
}

def carregar_cinza(caminho):
    rgb = np.array(Image.open(caminho).convert("RGB"), dtype=np.float64)
    cinza = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.round(cinza), 0, 255)

def media(x):
    return float(x.sum() / x.size)


def variancia(x):
    m = media(x)
    return float(((x - m) ** 2).sum() / x.size)


def desvio_padrao(x):
    return float(np.sqrt(variancia(x)))


def assimetria(x):
    """Skewness = E[(x-m)^3] / sigma^3."""
    m, s = media(x), desvio_padrao(x)
    if s == 0:
        return 0.0
    return float((((x - m) ** 3).sum() / x.size) / s ** 3)


def curtose(x):
    """Curtose (não-excesso) = E[(x-m)^4] / sigma^4  (normal = 3)."""
    m, s = media(x), desvio_padrao(x)
    if s == 0:
        return 0.0
    return float((((x - m) ** 4).sum() / x.size) / s ** 4)


def histograma_normalizado(x):
    contagem = np.zeros(256)
    for v in x.astype(np.int64).ravel():  # laço explícito evita np.histogram
        contagem[v] += 1
    return contagem / contagem.sum()


def entropia(x):
    """Entropia de Shannon (bits) do histograma de 256 níveis."""
    p = histograma_normalizado(x)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def energia(x):
    """Energia = soma dos quadrados das intensidades."""
    return float((x ** 2).sum())


def uniformidade(x):
    """Energia do histograma (soma de p^2): 1 = imagem constante, ~0 = histograma espalhado."""
    p = histograma_normalizado(x)
    return float((p ** 2).sum())


# ----------------------------------------------------------------------------
# Descritores estruturais simples
# ----------------------------------------------------------------------------
def dif_horizontal(x):
    return float(np.abs(x[:, 1:] - x[:, :-1]).sum() / (x.shape[0] * (x.shape[1] - 1)))


def dif_vertical(x):
    return float(np.abs(x[1:, :] - x[:-1, :]).sum() / ((x.shape[0] - 1) * x.shape[1]))


def dif_media_vizinhos(x):
    """Média das diferenças absolutas horizontais e verticais (todas as vizinhanças-4 em pares)."""
    dh = np.abs(x[:, 1:] - x[:, :-1])
    dv = np.abs(x[1:, :] - x[:-1, :])
    return float((dh.sum() + dv.sum()) / (dh.size + dv.size))


def desvio_local_blocos(x, n=8):
    """Desvio padrão de cada bloco n x n -> (média, % de blocos 'homogêneos' com desvio < 3)."""
    h, w = x.shape
    h8, w8 = (h // n) * n, (w // n) * n
    b = x[:h8, :w8].reshape(h8 // n, n, w8 // n, n).swapaxes(1, 2).reshape(-1, n * n)
    m = b.sum(axis=1) / (n * n)
    dp = np.sqrt(((b - m[:, None]) ** 2).sum(axis=1) / (n * n))
    return float(dp.sum() / dp.size), float((dp < 3).sum() / dp.size * 100)


def descritores(x):
    mn, mx = float(x.min()), float(x.max())
    dloc, homog = desvio_local_blocos(x)
    return {
        "media": media(x),
        "desvio_padrao": desvio_padrao(x),
        "variancia": variancia(x),
        "minimo": mn,
        "maximo": mx,
        "amplitude": mx - mn,
        "energia": energia(x),
        "entropia": entropia(x),
        "assimetria": assimetria(x),
        "curtose": curtose(x),
        "uniformidade": uniformidade(x),
        "dif_horizontal": dif_horizontal(x),
        "dif_vertical": dif_vertical(x),
        "dif_vizinhos": dif_media_vizinhos(x),
        "desvio_local_8x8": dloc,
        "blocos_homogeneos_pct": homog,
    }


def imprimir(titulo, d):
    print(f"--- {titulo} ---")
    print(f"Média de intensidade: {d['media']:.2f}  Desvio padrão: {d['desvio_padrao']:.2f}  "
          f"Variância: {d['variancia']:.2f}  Mínimo: {d['minimo']:.0f}")
    print(f"Máximo: {d['maximo']:.0f}  Amplitude: {d['amplitude']:.0f}  Energia: {d['energia']:.2e}  "
          f"Entropia: {d['entropia']:.2f}  Assimetria (Skewness): {d['assimetria']:.2f}  "
          f"Curtose: {d['curtose']:.2f}")
    print(f"Diferença média entre pixels vizinhos: {d['dif_vizinhos']:.2f} "
          f"(H: {d['dif_horizontal']:.2f} | V: {d['dif_vertical']:.2f})")
    print(f"Desvio local médio (blocos 8x8): {d['desvio_local_8x8']:.2f}  "
          f"Blocos homogêneos (desvio < 3): {d['blocos_homogeneos_pct']:.1f}%\n")


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def salvar_regioes(cinza, regioes, caminho):
    """Desenha os retângulos das regiões sobre a imagem (só NumPy + PIL para salvar)."""
    im = np.stack([cinza] * 3, axis=-1).astype(np.uint8)
    cores = [(255, 255, 255), (0, 0, 0), (255, 255, 255), (0, 0, 0)]
    for k, (_, (y0, y1, x0, x1)) in enumerate(regioes):
        c = cores[k % 4]
        for esp in range(3):
            im[y0 + esp, x0:x1] = c
            im[y1 - 1 - esp, x0:x1] = c
            im[y0:y1, x0 + esp] = c
            im[y0:y1, x1 - 1 - esp] = c
    Image.fromarray(im).save(caminho)


def main():
    resultados = {}
    for chave, (titulo, caminho) in IMAGENS.items():
        cinza = carregar_cinza(caminho)
        h, w = cinza.shape
        print(f"=== Análise da Imagem: {titulo} ===")
        print(f"Dimensões da imagem: {w} x {h}\n")
        salvar_regioes(cinza, REGIOES[chave], os.path.join(SAIDA, f"q2_{chave}_regioes.png"))

        res = {"titulo": titulo, "dim": [w, h], "regioes": {}}
        for k, (nome, (y0, y1, x0, x1)) in enumerate(REGIOES[chave]):
            reg = cinza[y0:y1, x0:x1]
            d = descritores(reg)
            d["caixa"] = [y0, y1, x0, x1]
            res["regioes"][nome] = d
            imprimir(f"Região {k + 1}: {nome}", d)
        dg = descritores(cinza)
        res["global"] = dg
        imprimir("Análise Global", dg)
        res["histograma"] = histograma_normalizado(cinza).tolist()
        resultados[chave] = res

    # --- Figura 1: histogramas globais ---
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    for ax, (chave, r) in zip(eixos, resultados.items()):
        ax.bar(np.arange(256), r["histograma"], width=1.0, color="0.35")
        ax.set_title(r["titulo"], fontsize=9)
        ax.set_xlabel("intensidade")
        ax.grid(alpha=0.3)
    eixos[0].set_ylabel("frequência relativa")
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA, "q2_histogramas.png"), dpi=170)
    plt.close()

    # --- Figura 2: comparação de descritores por região ---
    nomes, desv, ent, dif, hom, cores = [], [], [], [], [], []
    for chave, tom in (("ia", "0.2"), ("ilustracao", "0.6")):
        for nome, d in resultados[chave]["regioes"].items():
            nomes.append(("IA: " if chave == "ia" else "Ilust.: ") + nome)
            desv.append(d["desvio_padrao"])
            ent.append(d["entropia"])
            dif.append(d["dif_vizinhos"])
            hom.append(d["blocos_homogeneos_pct"])
            cores.append(tom)
    fig, eixos = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, vals, t in zip(eixos, (desv, ent, dif, hom),
                           ("Desvio padrão", "Entropia (bits)", "Dif. média vizinhos", "Blocos 8x8 homogêneos (%)")):
        ax.barh(np.arange(len(nomes)), vals, color=cores)
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    eixos[0].set_yticks(np.arange(len(nomes)))
    eixos[0].set_yticklabels(nomes, fontsize=8)
    eixos[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA, "q2_comparacao_regioes.png"), dpi=170)
    plt.close()

    with open(os.path.join(SAIDA, "q2_resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("Arquivos salvos em:", SAIDA)


if __name__ == "__main__":
    main()
