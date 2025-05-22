import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys
import traceback
from pose_analysis import process_image, process_camera, set_exercicio, exercicio_atual, get_reference_pose, create_stick_figure, pontuacao, etapa_atual
import cv2
import numpy as np
from split_screen_app import run_split_screen
from planos_treino import PlanoTreino, PLANOS_DISPONIVEIS

# Configurar handler de exceções
def exception_handler(exc_type, exc_value, exc_traceback):
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Erro não tratado:\n{error_msg}")
    messagebox.showerror("Erro", f"Ocorreu um erro inesperado:\n{str(exc_value)}")

sys.excepthook = exception_handler

INPUT_FOLDER = "input"

# Configuração de cores e estilos
COLORS = {
    "primary": "#1A6FA3",    # Azul escuro do mockup
    "secondary": "#174A6A",  # Azul dos botões
    "accent": "#FFD600",     # Amarelo do botão
    "background": "#3DDAD7", # Fundo azul esverdeado
    "text": "#FFFFFF",       # Texto branco
    "white": "#FFFFFF",
    "success": "#2ecc71",
    "warning": "#f1c40f",
    "error": "#e74c3c"
}

def create_button(parent, text, command, color=None, fg=None, font_size=16, bold=False):
    btn = tk.Button(parent,
                   text=text,
                   command=command,
                   font=("Helvetica", font_size, "bold" if bold else "normal"),
                   bg=color or COLORS["secondary"],
                   fg=fg or COLORS["white"],
                   activebackground=COLORS["primary"],
                   activeforeground=COLORS["white"],
                   relief=tk.FLAT,
                   padx=20,
                   pady=10,
                   cursor="hand2",
                   bd=0,
                   highlightthickness=0)
    return btn

def init_application():
    """Inicializa a aplicação com tratamento de erros"""
    try:
        root = tk.Tk()
        root.title("PoseFit")
        root.geometry("900x550")
        root.configure(bg=COLORS["background"])
        root.resizable(False, False)

        main_frame = tk.Frame(root, bg=COLORS["background"], padx=0, pady=0)
        main_frame.pack(expand=True, fill="both")

        # Esquerda: Exercícios
        left_frame = tk.Frame(main_frame, bg=COLORS["background"], width=400)
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))

        # Direita: Planos
        right_frame = tk.Frame(main_frame, bg=COLORS["background"], width=400)
        right_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=(10, 0))

        # Título Exercícios
        title_label_left = tk.Label(left_frame,
                            text="Exercícios Disponíveis",
                            font=("Helvetica", 20, "bold"),
                            fg=COLORS["text"],
                            bg=COLORS["background"])
        title_label_left.pack(pady=(10, 10))

        # Grid de botões de exercícios
        exercises = [
            "Agachamento", "Flexao", "Prancha",
            "Prancha", "Jumping Jacks", "Lunge",
            "Shoulder Press"
        ]
        grid_frame = tk.Frame(left_frame, bg=COLORS["background"])
        grid_frame.pack(pady=(0, 0))
        for i, exercise in enumerate(exercises):
            row = i // 2
            col = i % 2
            btn = create_button(
                grid_frame,
                exercise,
                lambda ex=exercise: start_exercise(ex),
                color=COLORS["secondary"],
                font_size=16,
                bold=True
            )
            btn.grid(row=row, column=col, padx=18, pady=12, sticky="ew")
        # Ajustar largura dos botões
        for col in range(2):
            grid_frame.grid_columnconfigure(col, weight=1)

        # Título Planos
        title_label_right = tk.Label(right_frame,
                            text="Planos de Treino",
                            font=("Helvetica", 20, "bold"),
                            fg=COLORS["text"],
                            bg=COLORS["background"])
        title_label_right.pack(pady=(10, 10))

        # Lista de planos
        plans_listbox = tk.Listbox(right_frame, 
                                font=("Helvetica", 14),
                                bg=COLORS["white"],
                                fg=COLORS["primary"],
                                selectbackground=COLORS["primary"],
                                selectforeground=COLORS["white"],
                                height=4,
                                bd=0,
                                highlightthickness=0)
        plans_listbox.pack(fill="x", padx=(0, 30), pady=(0, 10))
        for plan_name in PLANOS_DISPONIVEIS.keys():
            plans_listbox.insert(tk.END, plan_name)

        # Detalhes do plano
        plan_details_frame = tk.Frame(right_frame, bg=COLORS["background"])
        plan_details_frame.pack(fill="x", padx=(0, 30), pady=(0, 10))
        plan_details_label = tk.Label(plan_details_frame, text="", justify="left",
                                      font=("Helvetica", 14), bg=COLORS["white"], fg=COLORS["primary"], anchor="w",
                                      bd=0, relief=tk.FLAT, padx=20, pady=10, width=30)
        plan_details_label.pack(fill="x")

        def show_plan_details(event=None):
            selection = plans_listbox.curselection()
            if selection:
                plan_name = plans_listbox.get(selection[0])
                plan = PlanoTreino(plan_name, PLANOS_DISPONIVEIS[plan_name])
                details = f"Plano: {plan_name}\n"
                for i, exercise in enumerate(plan.exercicios, 1):
                    details += f"{i}. {exercise}\n"
                plan_details_label.config(text=details)
            else:
                plan_details_label.config(text="")
        plans_listbox.bind('<<ListboxSelect>>', show_plan_details)
        show_plan_details()  # Inicializa

        # Botão grande amarelo
        start_plan_button = create_button(
            right_frame,
            "Iniciar Plano Selecionado",
            lambda: start_selected_plan(),
            color=COLORS["accent"],
            fg=COLORS["primary"],
            font_size=20,
            bold=True
        )
        start_plan_button.pack(pady=(20, 0), fill="x", padx=(0, 30))

        # Logo PoseFit
        logo_label = tk.Label(left_frame, text="PoseFit", font=("Helvetica", 28, "bold"),
                              fg=COLORS["white"], bg=COLORS["background"])
        logo_label.pack(side=tk.BOTTOM, anchor="w", pady=10, padx=10)

        # Funções de ação (mantendo as originais)
        def start_exercise(exercise_name):
            try:
                tk.Tk.root = root
                root.withdraw()
                run_split_screen(exercise_name)
                root.deiconify()
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao iniciar exercício: {str(e)}")
                root.deiconify()

        def start_selected_plan():
            try:
                selection = plans_listbox.curselection()
                if selection:
                    plan_name = plans_listbox.get(selection[0])
                    plan = PlanoTreino(plan_name, PLANOS_DISPONIVEIS[plan_name])
                    tk.Tk.root = root
                    root.withdraw()
                    run_split_screen(plan.exercicio_atual(), plan)
                    root.deiconify()
                else:
                    messagebox.showwarning("Aviso", "Por favor, selecione um plano de treino.")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao iniciar plano: {str(e)}")
                root.deiconify()

        return root

    except Exception as e:
        messagebox.showerror("Erro de Inicialização", 
                           f"Erro ao inicializar a aplicação:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        root = init_application()
        root.mainloop()
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        sys.exit(1)
