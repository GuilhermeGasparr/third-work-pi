
import json
import os

import numpy as np
from PIL import Image  # somente load/save
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))
IMAGENS = {
    "ia": os.path.join(PASTA, "imagens", "imagem_ia_cinematografica.jpg"),
    "ilustracao": os.path.join(PASTA, "imagens", "imagem_ilustracao_digital.jpg"),
}
SAIDA = os.path.join(PASTA, "resultados")
os.makedirs(SAIDA, exist_ok=True)

N = 8  # tamanho do bloco

# Matriz de quantização de luminância do JPEG (Anexo K da norma) -
# usada como base da matriz "simplificada" (escalonada por um fator de qualidade).
Q_BASE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

QUALIDADES = [90, 50, 20, 10]


# ----------------------------------------------------------------------------
# Entrada / saída
# ----------------------------------------------------------------------------
def carregar_rgb(caminho):
    return np.array(Image.open(caminho).convert("RGB"), dtype=np.float64)


def salvar_cinza(matriz, caminho):
    """Salva matriz 2D (0..255) como PNG em tons de cinza."""
    Image.fromarray(np.clip(np.round(matriz), 0, 255).astype(np.uint8), mode="L").save(caminho)


# ----------------------------------------------------------------------------
# Pré-processamento manual
# ----------------------------------------------------------------------------
def rgb_para_cinza(rgb):
    """Luminância ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def preencher_multiplo(img, n=N):
    """Replica as bordas para que altura e largura sejam múltiplos de n."""
    h, w = img.shape
    ph, pw = (-h) % n, (-w) % n
    return np.pad(img, ((0, ph), (0, pw)), mode="edge")


def dividir_blocos(img, n=N):
    """(H, W) -> (nBlocosLin, nBlocosCol, n, n)."""
    h, w = img.shape
    return img.reshape(h // n, n, w // n, n).swapaxes(1, 2)


def juntar_blocos(blocos):
    """(nBL, nBC, n, n) -> (H, W)."""
    nbl, nbc, n, _ = blocos.shape
    return blocos.swapaxes(1, 2).reshape(nbl * n, nbc * n)


# ----------------------------------------------------------------------------
# DCT 2D manual
# ----------------------------------------------------------------------------
def matriz_dct(n=N):
    """
    Matriz ortonormal C da DCT-II 1D:
        C[u, x] = alpha(u) * cos((2x + 1) * u * pi / (2n))
        alpha(0) = sqrt(1/n),  alpha(u>0) = sqrt(2/n)
    A DCT 2D separável de um bloco B é  F = C . B . C^T
    e a inversa é                       B = C^T . F . C
    """
    C = np.zeros((n, n))
    for u in range(n):
        alpha = np.sqrt(1.0 / n) if u == 0 else np.sqrt(2.0 / n)
        for x in range(n):
            C[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return C


C = matriz_dct()


def dct2_bloco_ingenua(bloco):
    """Fórmula direta da DCT 2D (4 laços). Usada só para validar a versão rápida."""
    n = bloco.shape[0]
    F = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            au = np.sqrt(1.0 / n) if u == 0 else np.sqrt(2.0 / n)
            av = np.sqrt(1.0 / n) if v == 0 else np.sqrt(2.0 / n)
            soma = 0.0
            for x in range(n):
                for y in range(n):
                    soma += (bloco[x, y]
                             * np.cos((2 * x + 1) * u * np.pi / (2 * n))
                             * np.cos((2 * y + 1) * v * np.pi / (2 * n)))
            F[u, v] = au * av * soma
    return F


def dct2_blocos(blocos):
    """DCT 2D de todos os blocos ao mesmo tempo: F = C B C^T."""
    return C @ blocos @ C.T


def idct2_blocos(coefs):
    """IDCT 2D de todos os blocos: B = C^T F C."""
    return C.T @ coefs @ C


# ----------------------------------------------------------------------------
# Quantização
# ----------------------------------------------------------------------------
def matriz_quantizacao(qualidade):
    """Escalona Q_BASE pelo fator de qualidade (1..100), como no JPEG."""
    qualidade = max(1, min(100, qualidade))
    fator = 5000.0 / qualidade if qualidade < 50 else 200.0 - 2.0 * qualidade
    Q = np.floor((Q_BASE * fator + 50.0) / 100.0)
    return np.clip(Q, 1, 255)


def quantizar(coefs, Q):
    return np.round(coefs / Q)


def desquantizar(q, Q):
    return q * Q


# ----------------------------------------------------------------------------
# Métricas manuais
# ----------------------------------------------------------------------------
def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    e = mse(a, b)
    return float("inf") if e == 0 else float(10.0 * np.log10(255.0 ** 2 / e))


def medida_blocagem(img, n=N):
    """
    Razão entre a diferença média absoluta de pixels vizinhos que cruzam a
    fronteira dos blocos e a dos que estão dentro do bloco.
    Valores > 1 indicam artefato de bloco visível (descontinuidade nas bordas).
    """
    h, w = img.shape
    dh = np.abs(np.diff(img, axis=1))  # entre colunas j e j+1
    dv = np.abs(np.diff(img, axis=0))
    colunas = np.arange(dh.shape[1])
    linhas = np.arange(dv.shape[0])
    borda_h = (colunas % n) == (n - 1)
    borda_v = (linhas % n) == (n - 1)
    fora = np.concatenate([dh[:, borda_h].ravel(), dv[borda_v, :].ravel()])
    dentro = np.concatenate([dh[:, ~borda_h].ravel(), dv[~borda_v, :].ravel()])
    return float(fora.mean() / dentro.mean())


def entropia_coeficientes(q):
    """Entropia (bits/coef.) dos coeficientes quantizados - estimativa do custo de codificação."""
    valores, contagens = np.unique(q.astype(np.int64), return_counts=True)
    p = contagens / contagens.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------------
# Pipeline completo
# ----------------------------------------------------------------------------
def comprimir_reconstruir(cinza, qualidade):
    h, w = cinza.shape
    base = preencher_multiplo(cinza) - 128.0          # deslocamento de nível
    blocos = dividir_blocos(base)
    coefs = dct2_blocos(blocos)
    Q = matriz_quantizacao(qualidade)
    q = quantizar(coefs, Q)
    rec_blocos = idct2_blocos(desquantizar(q, Q))
    rec = juntar_blocos(rec_blocos) + 128.0
    rec = np.clip(np.round(rec), 0, 255)[:h, :w]
    return rec, coefs, q, Q


def montar_painel(imagens, separador=8):
    """Concatena imagens 2D lado a lado (com faixa branca) - só NumPy."""
    h = max(i.shape[0] for i in imagens)
    colunas = []
    for k, im in enumerate(imagens):
        if k > 0:
            colunas.append(np.full((h, separador), 255.0))
        colunas.append(np.pad(im, ((0, h - im.shape[0]), (0, 0)), constant_values=255))
    return np.concatenate(colunas, axis=1)


def ampliar(img, fator):
    """Ampliação por vizinho mais próximo (np.repeat) para ver os blocos."""
    return np.repeat(np.repeat(img, fator, axis=0), fator, axis=1)


def validar_dct():
    rng = np.random.default_rng(0)
    bloco = rng.integers(0, 256, (N, N)).astype(float) - 128
    erro_dct = np.abs(dct2_bloco_ingenua(bloco) - (C @ bloco @ C.T)).max()
    erro_inv = np.abs(C.T @ (C @ bloco @ C.T) @ C - bloco).max()
    return float(erro_dct), float(erro_inv)


def main():
    erro_dct, erro_inv = validar_dct()
    print(f"Validação da DCT: |rápida - fórmula direta| = {erro_dct:.2e}; "
          f"|IDCT(DCT(B)) - B| = {erro_inv:.2e}")

    resultados = {"validacao": {"erro_dct": erro_dct, "erro_inversa": erro_inv},
                  "qualidades": QUALIDADES, "imagens": {}}

    for chave, caminho in IMAGENS.items():
        cinza = rgb_para_cinza(carregar_rgb(caminho))
        h, w = cinza.shape
        salvar_cinza(cinza, os.path.join(SAIDA, f"q1_{chave}_cinza.png"))
        print(f"\n=== {chave} ({w}x{h}) ===")
        res = {"dim": [w, h], "por_qualidade": {}}
        recs = {}
        base_cinza_pad = preencher_multiplo(cinza)
        res["blocagem_original"] = medida_blocagem(cinza)

        for qual in QUALIDADES:
            rec, coefs, q, Q = comprimir_reconstruir(cinza, qual)
            recs[qual] = rec
            salvar_cinza(rec, os.path.join(SAIDA, f"q1_{chave}_rec_q{qual}.png"))
            erro = np.abs(cinza - rec)
            salvar_cinza(np.clip(erro * 8, 0, 255), os.path.join(SAIDA, f"q1_{chave}_erro_q{qual}.png"))
            zeros = float((q == 0).mean() * 100)
            # energia relativa dos coeficientes de alta frequência (u+v >= 8) antes da quantização
            u, v = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
            mask_alta = (u + v) >= 8
            e_total = float((coefs[..., 1:, 1:] ** 2).sum() + 1e-12)
            e_alta = float((coefs[..., mask_alta] ** 2).sum())
            res["por_qualidade"][str(qual)] = {
                "mse": mse(cinza, rec),
                "psnr": psnr(cinza, rec),
                "zeros_pct": zeros,
                "entropia_bits": entropia_coeficientes(q),
                "blocagem": medida_blocagem(rec),
                "erro_max": float(erro.max()),
            }
            print(f"  Q={qual:3d}  MSE={mse(cinza, rec):8.2f}  PSNR={psnr(cinza, rec):6.2f} dB  "
                  f"coef.=0: {zeros:5.1f}%  H={entropia_coeficientes(q):.2f} bits  "
                  f"blocagem={medida_blocagem(rec):.3f}")

        # energia de alta frequência (independe da qualidade)
        _, coefs0, _, _ = comprimir_reconstruir(cinza, 50)
        u, v = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        ac = (u + v) > 0
        alta = (u + v) >= 6
        res["energia_alta_freq_pct"] = float((coefs0[..., alta] ** 2).sum() / (coefs0[..., ac] ** 2).sum() * 100)
        res["media_ac_abs"] = float(np.abs(coefs0[..., ac]).mean())

        # Painel completo: original | Q=90 | Q=50 | Q=20 | Q=10
        painel = montar_painel([cinza] + [recs[qq] for qq in QUALIDADES])
        salvar_cinza(painel, os.path.join(SAIDA, f"q1_{chave}_painel.png"))

        # Zoom (recorte 64x64 ampliado 5x) para evidenciar blocos
        if chave == "ia":
            y0, x0 = 130, 760   # olho / região do rosto
        else:
            y0, x0 = 260, 600   # rosto da personagem / chapéu
        recorte = lambda im: ampliar(im[y0:y0 + 64, x0:x0 + 64], 5)
        zoom = montar_painel([recorte(cinza)] + [recorte(recs[qq]) for qq in (50, 20, 10)])
        salvar_cinza(zoom, os.path.join(SAIDA, f"q1_{chave}_zoom.png"))
        res["zoom_origem"] = [y0, x0]
        resultados["imagens"][chave] = res

    # Gráfico de métricas
    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.3))
    estilos = {"ia": ("o-", "0.15", "Imagem 1 (IA)"), "ilustracao": ("s--", "0.5", "Imagem 2 (ilustração)")}
    for chave, (mk, cor, nome) in estilos.items():
        pq = resultados["imagens"][chave]["por_qualidade"]
        x = QUALIDADES
        eixos[0].plot(x, [pq[str(k)]["psnr"] for k in x], mk, color=cor, label=nome)
        eixos[1].plot(x, [pq[str(k)]["zeros_pct"] for k in x], mk, color=cor, label=nome)
        eixos[2].plot(x, [pq[str(k)]["blocagem"] for k in x], mk, color=cor, label=nome)
    for ax, t, yl in zip(eixos, ["PSNR", "Coeficientes zerados", "Medida de blocagem"],
                         ["dB", "%", "razão borda/interior"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("fator de qualidade")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    eixos[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA, "q1_metricas.png"), dpi=170)
    plt.close()

    with open(os.path.join(SAIDA, "q1_resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\nArquivos salvos em:", SAIDA)


if __name__ == "__main__":
    main()
