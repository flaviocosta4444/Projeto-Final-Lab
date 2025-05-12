import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from pose_analysis import process_image, process_camera, set_exercicio, exercicio_atual

INPUT_FOLDER = "input"

# Configuração de cores e estilos
COLORS = {
    "primary": "#2C3E50",    # Azul escuro
    "secondary": "#3498DB",  # Azul claro
    "accent": "#E74C3C",     # Vermelho
    "background": "#ECF0F1", # Cinza claro
    "text": "#2C3E50",       # Texto escuro
    "white": "#FFFFFF",      # Branco
    "success": "#2ecc71",    # Verde sucesso
    "warning": "#f1c40f",    # Amarelo aviso
    "error": "#e74c3c"       # Vermelho erro
}

# Janela principal
root = tk.Tk()
root.title("Análise de Pose - Sistema de Exercícios")
root.geometry("800x600")
root.configure(bg=COLORS["background"])

# Estilo para os widgets
style = ttk.Style()
style.configure("Custom.TButton",
                padding=10,
                font=("Helvetica", 12),
                background=COLORS["secondary"])

# Frame principal
main_frame = tk.Frame(root, bg=COLORS["background"], padx=20, pady=20)
main_frame.pack(expand=True, fill="both")

# Título
title_frame = tk.Frame(main_frame, bg=COLORS["background"])
title_frame.pack(pady=(0, 20))

title_label = tk.Label(title_frame,
                      text="Sistema de Análise de Exercícios",
                      font=("Helvetica", 24, "bold"),
                      fg=COLORS["primary"],
                      bg=COLORS["background"])
title_label.pack()

subtitle_label = tk.Label(title_frame,
                         text="Siga o modelo para realizar o exercício corretamente",
                         font=("Helvetica", 14),
                         fg=COLORS["text"],
                         bg=COLORS["background"])
subtitle_label.pack(pady=(5, 0))

# Descrição da nova funcionalidade
description_frame = tk.Frame(main_frame, bg=COLORS["background"])
description_frame.pack(pady=20)

description_text = """
Este sistema agora mostra um modelo 3D de referência em vista lateral que você deve seguir.
Na tela dividida, você verá sua pose à esquerda e o modelo 3D a seguir à direita.
Tente imitar a pose do modelo para obter uma pontuação alta!
O modelo 3D proporciona uma melhor visualização da forma correta do exercício.
"""

description_label = tk.Label(description_frame,
                            text=description_text,
                            font=("Helvetica", 12),
                            fg=COLORS["text"],
                            bg=COLORS["background"],
                            justify=tk.LEFT,
                            wraplength=600)
description_label.pack()

# Function to create styled buttons
def create_button(parent, text, command):
    btn = tk.Button(parent,
                   text=text,
                   command=command,
                   font=("Helvetica", 12),
                   bg=COLORS["secondary"],
                   fg=COLORS["white"],
                   activebackground=COLORS["primary"],
                   activeforeground=COLORS["white"],
                   relief=tk.FLAT,
                   padx=20,
                   pady=10,
                   cursor="hand2")
    return btn

# Frame para os botões de exercícios
exercise_buttons_frame = tk.Frame(main_frame, bg=COLORS["background"])
exercise_buttons_frame.pack(pady=20)

# Título para os botões de exercícios
tk.Label(exercise_buttons_frame,
        text="Selecione um Exercício para Iniciar:",
        font=("Helvetica", 14, "bold"),
        fg=COLORS["primary"],
        bg=COLORS["background"]).pack(pady=(0, 10))

# Grid para organizar os botões de exercícios
buttons_grid = tk.Frame(exercise_buttons_frame, bg=COLORS["background"])
buttons_grid.pack()

# Funções para iniciar exercícios
def start_exercise(exercise_name):
    set_exercicio(exercise_name)
    process_camera()

# Adicionar botões para cada exercício em uma grade de 2x3
exercises = [
    "Agachamento", "Flexao", "Prancha",
    "Lunge", "Deadlift", "Shoulder Press"
]

for i, exercise in enumerate(exercises):
    row = i // 3
    col = i % 3
    
    # Frame para cada botão e seu ícone
    exercise_frame = tk.Frame(buttons_grid, bg=COLORS["background"])
    exercise_frame.grid(row=row, column=col, padx=15, pady=15)
    
    # Botão para iniciar o exercício
    btn = create_button(
        exercise_frame,
        exercise,
        lambda ex=exercise: start_exercise(ex)
    )
    btn.pack()

# Frame para os botões de análise de imagem
image_analysis_frame = tk.Frame(main_frame, bg=COLORS["background"])
image_analysis_frame.pack(pady=20)

def process_image_gui():
    # Abrir diálogo para selecionar exercício primeiro
    exercise_select = tk.Toplevel(root)
    exercise_select.title("Selecione o Exercício")
    exercise_select.geometry("300x300")
    exercise_select.configure(bg=COLORS["background"])
    
    tk.Label(exercise_select,
            text="Selecione o exercício para análise:",
            font=("Helvetica", 12),
            fg=COLORS["text"],
            bg=COLORS["background"]).pack(pady=10)
    
    for exercise in exercises:
        btn = tk.Button(
            exercise_select,
            text=exercise,
            command=lambda ex=exercise: select_and_open_image(ex, exercise_select),
            font=("Helvetica", 12),
            bg=COLORS["secondary"],
            fg=COLORS["white"],
            activebackground=COLORS["primary"],
            activeforeground=COLORS["white"],
            relief=tk.FLAT,
            padx=20,
            pady=5,
            cursor="hand2"
        )
        btn.pack(pady=5)

def select_and_open_image(selected_exercise, window):
    window.destroy()
    set_exercicio(selected_exercise)
    
    if not os.path.exists(INPUT_FOLDER):
        messagebox.showerror("Erro", f"A pasta '{INPUT_FOLDER}' não existe.")
        return

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        messagebox.showinfo("Sem imagens", f"Nenhuma imagem encontrada na pasta '{INPUT_FOLDER}'.")
        return

    top = tk.Toplevel(root)
    top.title("Seleciona uma imagem")
    top.configure(bg=COLORS["background"])

    # Frame para o título
    title_frame = tk.Frame(top, bg=COLORS["background"])
    title_frame.pack(pady=10)
    
    tk.Label(title_frame,
            text=f"Selecione uma imagem para análise de {selected_exercise}",
            font=("Helvetica", 14),
            fg=COLORS["primary"],
            bg=COLORS["background"]).pack()

    # Frame para o canvas e scrollbar
    canvas_frame = tk.Frame(top, bg=COLORS["background"])
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    canvas = tk.Canvas(canvas_frame, bg=COLORS["background"])
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(canvas_frame, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas, bg=COLORS["background"])
    canvas.create_window((0, 0), window=frame, anchor='nw')

    def on_image_click(image_path):
        top.destroy()
        process_image(image_path)

    # Grid para as imagens
    row = 0
    col = 0
    for img_name in images:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        try:
            pil_image = Image.open(img_path).resize((200, 200))
            photo = ImageTk.PhotoImage(pil_image)
            
            # Frame para cada imagem
            img_frame = tk.Frame(frame, bg=COLORS["background"])
            img_frame.grid(row=row, column=col, padx=10, pady=10)
            
            img_btn = tk.Button(img_frame,
                              image=photo,
                              command=lambda p=img_path: on_image_click(p),
                              relief=tk.FLAT,
                              cursor="hand2")
            img_btn.image = photo
            img_btn.pack()
            
            # Nome do arquivo
            tk.Label(img_frame,
                    text=img_name,
                    font=("Helvetica", 10),
                    fg=COLORS["text"],
                    bg=COLORS["background"]).pack()
            
            col += 1
            if col > 2:  # 3 imagens por linha
                col = 0
                row += 1
                
        except Exception as e:
            print(f"[ERRO] Não foi possível carregar a imagem {img_path}: {e}")

    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

# Botão para análise de imagem
btn_image = create_button(image_analysis_frame, "Analisar Imagem", process_image_gui)
btn_image.pack()

# Footer
footer_frame = tk.Frame(main_frame, bg=COLORS["background"])
footer_frame.pack(side=tk.BOTTOM, pady=20)

tk.Label(footer_frame,
        text="Sistema de Análise de Exercícios © 2025",
        font=("Helvetica", 10),
        fg=COLORS["text"],
        bg=COLORS["background"]).pack()

if __name__ == "__main__":
    root.mainloop()