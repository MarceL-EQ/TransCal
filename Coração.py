import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar
from scipy.special import j0, j1, jn_zeros

# ==============================================================================
# SEÇÃO DE ENTRADAS DO USUÁRIO
# ENTRADAS PARA O PROBLEMA DO CORAÇÃO
# ==============================================================================

# --- Propriedades do Material e Condições Iniciais ---
alpha = 1.4e-7      # Difusividade térmica do tecido cardíaco [m^2/s]
k = 1.0             # Condutividade térmica [W/m.K] (valor simbólico, não usado quando h=inf)
T1 = 37.0           # Temperatura inicial do coração [°C]
T_inf = 0.0         # Temperatura da solução salina [°C]

# --- Geometria do Cilindro ---
R = 0.0375          # Raio do cilindro (coração) [m]
L = 0.09            # Altura (comprimento) do cilindro [m]

# --- Condição de Convecção ---
# Superfície mantida a T_inf significa convecção "infinita"
h = np.inf

# --- Tempo e Ponto de Análise ---
t = 10800.0         # Tempo de 3 horas em segundos
r_ponto = 0.0       # Coordenada radial do centro
z_ponto = L / 2     # Coordenada axial do centro

# --- Parâmetros de Precisão da Solução ---
N_termos_radial = 20
N_termos_axial = 20

# ==============================================================================
# IMPLEMENTAÇÃO DA SOLUÇÃO ANALÍTICA (NÃO ALTERAR ESTA SEÇÃO)
# ==============================================================================

def encontrar_autovalores(Bi_r, Bi_z, N_r, N_z):
    """Encontra os autovalores (raízes) para as direções radial e axial."""
    if np.isinf(Bi_r):
        lambda_n = jn_zeros(0, N_r)
    else:
        def eq_radial(lam):
            return lam * j1(lam) - Bi_r * j0(lam)
        lambda_n = []
        for i in range(1, N_r + 1):
            guess = (i - 0.5) * np.pi
            try:
                sol = root_scalar(eq_radial, bracket=[guess, guess + np.pi / 2], method='brentq')
                lambda_n.append(sol.root)
            except ValueError:
                sol = root_scalar(eq_radial, bracket=[0.1, (i + 1) * np.pi], method='brentq')
                if sol.root not in lambda_n:
                    lambda_n.append(sol.root)
    if np.isinf(Bi_z):
        beta_n = [(i * np.pi) for i in range(1, N_z + 1)]
    else:
        def eq_axial(bet):
            return np.sin(bet) * (bet ** 2 - Bi_z ** 2) - np.cos(bet) * (2 * bet * Bi_z)
        beta_n = []
        for i in range(1, N_z + 1):
            guess = (i - 0.5) * np.pi
            sol = fsolve(eq_axial, guess)
            beta_n.append(sol[0])
    return np.array(lambda_n), np.array(beta_n)

def calcular_coeficientes(lambda_vals, beta_vals, Bi_z):
    """Calcula os coeficientes da série dupla C_mn."""
    C_mn = np.zeros((len(lambda_vals), len(beta_vals)))
    for m, lam in enumerate(lambda_vals):
        for n, bet in enumerate(beta_vals):
            num_int_r = j1(lam) / lam
            if np.isinf(Bi_z):
                num_int_z = (1 - np.cos(bet)) / bet
            else:
                num_int_z = (np.sin(bet) + (Bi_z / bet) * (1 - np.cos(bet)))
            den_int_r = 0.5 * (j0(lam) ** 2 + j1(lam) ** 2)
            if np.isinf(Bi_z):
                den_int_z = 0.5
            else:
                den_int_z = (bet ** 2 + Bi_z ** 2) * (1 + np.sin(2 * bet) / (2 * bet)) / (2 * bet ** 2) + Bi_z / (2 * bet ** 2) * (np.sin(bet) ** 2)
            C_mn[m, n] = (num_int_r * num_int_z) / (den_int_r * den_int_z)
    return C_mn

def calcular_temperatura(r, z, t, R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, Bi_z):
    """Calcula a temperatura no ponto (r, z) no tempo t usando a série dupla."""
    theta = 0.0
    r_star = r / R
    z_star = z / L
    for m, lam in enumerate(lambda_vals):
        for n, bet in enumerate(beta_vals):
            P_m = j0(lam * r_star)
            if np.isinf(Bi_z):
                Z_n = np.sin(bet * z_star)
            else:
                Z_n = np.cos(bet * z_star) + (Bi_z / bet) * np.sin(bet * z_star)
            expoente = -(lam ** 2 + (R / L) ** 2 * bet ** 2) * (alpha * t / R ** 2)
            G_mn = np.exp(expoente)
            theta += C_mn[m, n] * P_m * Z_n * G_mn
    T = T_inf + (T1 - T_inf) * theta
    return T

def gerar_graficos(R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, t, Bi_z):
    """Gera o mapa de calor e gráficos de distribuição de temperatura."""
    r_vals = np.linspace(0, R, 50)
    z_vals = np.linspace(0, L, 100)
    r_grid, z_grid = np.meshgrid(r_vals, z_vals)
    T_grid = np.zeros(r_grid.shape)
    print("\nCalculando grade para o mapa de calor (pode levar um momento)...")
    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            T_grid[i, j] = calcular_temperatura(r_grid[i, j], z_grid[i, j], t, R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, Bi_z)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # Define os níveis do mapa de cor para melhor visualização
    niveis = np.linspace(T_inf, T1, 25)
    cp = plt.contourf(r_grid, z_grid, T_grid, levels=20, cmap='coolwarm', vmin=T_inf)
    plt.colorbar(cp, label='Temperatura [°C]')
    plt.title(f'Mapa de Temperatura em t = {t/3600:.1f} horas')
    plt.xlabel('Raio (r) [m]')
    plt.ylabel('Altura (z) [m]')
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    T_radial = [calcular_temperatura(r, L / 2, t, R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, Bi_z) for r in r_vals]
    plt.plot(r_vals, T_radial, label=f'T(r, z=L/2)')
    T_axial = [calcular_temperatura(0, z, t, R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, Bi_z) for z in z_vals]
    plt.plot(z_vals, T_axial, label=f'T(r=0, z)', linestyle='--')
    plt.title(f'Distribuição de Temperatura em t = {t/3600:.1f} h')
    plt.xlabel('Posição [m]')
    plt.ylabel('Temperatura [°C]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    print("Iniciando a simulação do resfriamento do coração...")
    print(f"Dados: R={R*100}cm, L={L*100}cm, t={t/3600}h, h={h} W/m^2.K")
    if np.isinf(h):
        Bi_r = np.inf
        Bi_z = np.inf
    else:
        Bi_r = (h * R) / k
        Bi_z = (h * L) / k
    print(f"Encontrando {N_termos_radial} autovalores radiais e {N_termos_axial} axiais...")
    lambda_vals, beta_vals = encontrar_autovalores(Bi_r, Bi_z, N_termos_radial, N_termos_axial)
    print("Calculando os coeficientes da série...")
    C_mn = calcular_coeficientes(lambda_vals, beta_vals, Bi_z)
    print(f"Calculando a temperatura no ponto central (r={r_ponto}, z={z_ponto})...")
    T_final = calcular_temperatura(r_ponto, z_ponto, t, R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, Bi_z)
    print("\n--- RESULTADO FINAL ---")
    print(f"A temperatura no centro do coração após {t/3600:.1f} horas é: {T_final:.4f} °C")
    gerar_graficos(R, L, alpha, T1, T_inf, lambda_vals, beta_vals, C_mn, t, Bi_z)
