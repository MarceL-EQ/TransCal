import numpy as np
from scipy.special import i0, i1, k0, k1
import matplotlib.pyplot as plt

def calcular_temperatura_disco(To, Ro, Ri, delta, Ra, h, T_inf, k):
    """
    Calcula a distribuição de temperatura em um disco com duas regiões.

    Parâmetros:
    To (float): Temperatura da tubulação (em r=Ri) [°C]
    Ro (float): Raio externo do disco [m]
    Ri (float): Raio interno do disco (raio da tubulação) [m]
    delta (float): Espessura do disco [m]
    Ra (float): Raio até onde o disco está isolado nas faces [m]
    h (float): Coeficiente de transferência de calor por convecção [W/(m².°C)]
    T_inf (float): Temperatura ambiente [°C]
    k (float): Condutividade térmica do material do disco [W/(m.°C)]

    Retorna:
    funcao_temperatura (function): Uma função que recebe 'r' e retorna T(r).
    """

    # Validar as entradas dos raios
    if not (0 < Ri < Ra < Ro):
        raise ValueError("Os raios devem satisfazer 0 < Ri < Ra < Ro para esta configuração de duas regiões.")
    if delta <= 0:
        raise ValueError("A espessura do disco (delta) deve ser maior que zero.")
    if h < 0:
        raise ValueError("O coeficiente de convecção (h) não pode ser negativo.")
    if k <= 0:
        raise ValueError("A condutividade térmica (k) deve ser maior que zero.")

    # Calcular m
    m = np.sqrt((2 * h) / (k * delta))

    # --- Calcular os Termos Auxiliares ---
    # Term_1 = I_0(mR_a) + (I_1(mR_o) / K_1(mR_o)) * K_0(mR_a)
    # Term_2 = I_1(mR_a) - (I_1(mR_o) / K_1(mR_o)) * K_1(mR_a)

    # Lidando com o caso onde K_1(mRo) pode ser muito pequeno (próximo de zero)
    # Isso pode acontecer para mRo muito grande. Podemos verificar e adicionar uma pequena segurança.
    if k1(m * Ro) == 0:
        # Se K_1(mRo) é zero, a divisão seria por zero.
        # Para mRo grande, K_1(mRo) se aproxima de zero. Pode indicar uma aleta infinita
        # mas no contexto de um problema finito com ponta isolada, deve ter um valor.
        # Um valor muito pequeno pode causar instabilidade numérica.
        # Uma alternativa seria usar um pequeno epsilon ou um tratamento assintótico se o problema permitisse.
        # Por simplicidade, vamos levantar um erro para alertar sobre valores extremos.
        raise ZeroDivisionError(f"K_1(mR_o) é zero ou muito próximo de zero ({k1(m*Ro)}). Verifique os parâmetros h, k, delta, Ro.")
    
    Term_1 = i0(m * Ra) + (i1(m * Ro) / k1(m * Ro)) * k0(m * Ra)
    
    # CORREÇÃO APLICADA AQUI: O termo final era k1(m * Ro) por engano no código anterior.
    # A fórmula correta para Term_2 exige k1(m * Ra).
    Term_2 = i1(m * Ra) - (i1(m * Ro) / k1(m * Ro)) * k1(m * Ra) 

    # --- Calcular as Constantes C_C, C_D, C_A, C_B ---

    # Denominador comum para C_C
    denominador_Cc = (Ra * m * Term_2 * np.log(Ra / Ri)) - Term_1

    if denominador_Cc == 0:
        raise ZeroDivisionError("O denominador para C_C é zero. Isso pode indicar um problema com as condições de contorno ou um caso degenerado.")

    Cc = (T_inf - To) / denominador_Cc
    Cd = Cc * (i1(m * Ro) / k1(m * Ro))

    # C_A
    Ca = (T_inf - To + Cc * Term_1) / np.log(Ra / Ri)

    # C_B
    Cb = To - Ca * np.log(Ri)

    # --- Definir a Função de Temperatura T(r) ---
    def T_r(r_val):
        # r_val pode ser um escalar ou um array numpy
        r_val_array = np.asarray(r_val, dtype=float) # Garante que r_val é um array para operações vetorizadas

        # Inicializa o array de resultados com NaN (Not a Number)
        T_vals = np.full_like(r_val_array, np.nan, dtype=float)

        # Região 1: Isolada (Ri <= r <= Ra)
        # np.log(0) resultaria em infinito, então excluímos r=0 se Ri for 0.
        # No nosso problema, Ri > 0, então log(r) é bem definido.
        mask1 = (r_val_array >= Ri) & (r_val_array <= Ra)
        T_vals[mask1] = Ca * np.log(r_val_array[mask1]) + Cb

        # Região 2: Com convecção (Ra < r <= Ro)
        mask2 = (r_val_array > Ra) & (r_val_array <= Ro)
        T_vals[mask2] = T_inf + Cc * i0(m * r_val_array[mask2]) + Cd * k0(m * r_val_array[mask2])
        
        return T_vals

    return T_r

def main():
    print("--- Calculadora e Gerador de Gráfico de Temperatura do Disco ---")
    print("Considera duas regiões: isolada (Ri a Ra) e com convecção (Ra a Ro).")

    # Entrada de dados do usuário
    while True:
        try:
            To = float(input("Digite a temperatura da tubulação, To (°C): "))
            Ro = float(input("Digite o raio externo do disco, Ro (m): "))
            Ri = float(input("Digite o raio interno do disco (tubulação), Ri (m): "))
            delta = float(input("Digite a espessura do disco, delta (m): "))
            Ra = float(input("Digite o raio até onde o disco está isolado nas faces, Ra (m): "))
            h = float(input("Digite o coeficiente de transferência de calor por convecção, h (W/(m².°C)): "))
            T_inf = float(input("Digite a temperatura ambiente, T_inf (°C): "))
            
            # Escolha um K qualquer para a aleta (condutividade térmica)
            k = float(input("Digite a condutividade térmica do material, k (W/(m.°C), e.g., 200 para alumínio): "))
            
            # Validação dos raios
            if not (0 < Ri < Ra < Ro):
                print("\nErro: Os raios devem satisfazer 0 < Ri < Ra < Ro. Tente novamente.")
                continue
            break
        except ValueError:
            print("\nEntrada inválida. Por favor, digite um número para todos os parâmetros.")
        except Exception as e:
            print(f"\nOcorreu um erro inesperado na entrada de dados: {e}. Tente novamente.")

    try:
        # Gerar a função de temperatura
        T_func = calcular_temperatura_disco(To, Ro, Ri, delta, Ra, h, T_inf, k)

        # 1. Calcular a temperatura em um 'r' fornecido pelo usuário
        while True:
            try:
                r_input = input(f"\nDigite um raio r para calcular a temperatura [{Ri:.4f} a {Ro:.4f}] (ou 'q' para sair e gerar os gráficos): ")
                if r_input.lower() == 'q':
                    break
                r_user = float(r_input)
                
                # Para evitar erro de fora do range da função T_r, filtramos aqui.
                if r_user < Ri or r_user > Ro:
                    print(f"Erro: Raio {r_user:.4f} fora do intervalo válido [{Ri:.4f}, {Ro:.4f}].")
                else:
                    temp_at_r = T_func(r_user)
                    print(f"A temperatura em r = {r_user:.4f} m é: {temp_at_r:.2f} °C")
            except ValueError:
                print("Entrada inválida. Por favor, digite um número ou 'q'.")
            except Exception as e:
                print(f"Ocorreu um erro ao calcular a temperatura: {e}")
                
        # 2. Gerar o gráfico de linha da distribuição de temperatura
        num_points_line_plot = 200 # Número de pontos para o gráfico de linha
        r_values_line_plot = np.linspace(Ri, Ro, num_points_line_plot)
        temperatures_line_plot = T_func(r_values_line_plot)

        plt.figure(figsize=(12, 6)) # Aumenta a figura para dois subplots
        plt.subplot(1, 2, 1) # Define o primeiro subplot para o gráfico de linha
        plt.plot(r_values_line_plot, temperatures_line_plot, label='Distribuição de Temperatura T(r)')
        plt.axvline(x=Ra, color='r', linestyle='--', label=f'Interface (Ra = {Ra:.3f} m)')
        plt.title('Distribuição de Temperatura Radial')
        plt.xlabel('Raio, r (m)')
        plt.ylabel('Temperatura, T (°C)')
        plt.grid(True)
        plt.legend()


        # 3. Gerar o mapa de temperaturas do disco (heatmap circular)
        num_grid_points = 150 # Número de pontos na grade para o mapa (maior = mais detalhe, mais lento)
        x = np.linspace(-Ro, Ro, num_grid_points)
        y = np.linspace(-Ro, Ro, num_grid_points)
        X, Y = np.meshgrid(x, y) # Cria uma grade 2D de coordenadas x e y
        R_grid = np.sqrt(X**2 + Y**2) # Calcula o raio para cada ponto (x,y) na grade

        # Calcular temperaturas para a grade
        # Inicializa a grade de temperaturas com NaN. Pontos fora do disco ou no furo ficarão NaN.
        T_grid = np.full_like(R_grid, np.nan) 
        
        # Máscara para pontos que estão DENTRO do anel do disco (Ri <= r <= Ro)
        disk_mask = (R_grid >= Ri) & (R_grid <= Ro)
        
        # Aplica a função de temperatura T_func apenas nos pontos dentro do disco.
        # A função T_func já lida com as duas regiões (isolada/convecção) internamente.
        T_grid[disk_mask] = T_func(R_grid[disk_mask])

        plt.subplot(1, 2, 2) # Define o segundo subplot para o mapa de temperatura
        
        # Usar pcolormesh para plotar o mapa de calor.
        # A opção 'shading='auto'' lida melhor com grids não uniformes ou irregulares.
        # vmin/vmax ajudam a fixar a escala de cores, tornando o mapa mais consistente.
        cmap = plt.get_cmap('jet') # Escolha um mapa de cores (ex: 'jet', 'viridis', 'plasma', 'hot', 'cool')
        im = plt.pcolormesh(X, Y, T_grid, cmap=cmap, shading='auto', vmin=T_inf, vmax=To) 
        
        plt.colorbar(im, label='Temperatura (°C)') # Adiciona barra de cores
        plt.title('Mapa de Temperatura do Disco')
        plt.xlabel('Eixo X (m)')
        plt.ylabel('Eixo Y (m)')
        # Garante que os eixos tenham a mesma proporção, para o disco parecer circular
        plt.gca().set_aspect('equal', adjustable='box') 
        plt.xlim(-Ro, Ro) # Limita os eixos para o tamanho do disco
        plt.ylim(-Ro, Ro)

        # Desenha círculos para Ri e Ra para melhor visualização das regiões
        circle_Ri = plt.Circle((0, 0), Ri, color='black', fill=False, linestyle='--', linewidth=1, label='Raio Interno (Ri)')
        circle_Ra = plt.Circle((0, 0), Ra, color='gray', fill=False, linestyle=':', linewidth=1, label='Interface (Ra)')
        plt.gca().add_patch(circle_Ri)
        plt.gca().add_patch(circle_Ra)
        
        plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos
        plt.show() # Mostra as janelas dos gráficos

    except Exception as e:
        print(f"\nOcorreu um erro crítico durante o cálculo ou plotagem: {e}")
        import traceback
        traceback.print_exc() # Imprime o stack trace completo para depuração

if __name__ == "__main__":
    main()