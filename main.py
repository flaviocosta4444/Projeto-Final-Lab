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

def create_button(parent, text, command):
    """Função para criar botões estilizados"""
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

def init_application():
    """Inicializa a aplicação com tratamento de erros"""
    try:
        # Janela principal
        root = tk.Tk()
        root.title("Análise de Pose - Sistema de Exercícios")
        root.geometry("1200x800")
        root.configure(bg=COLORS["background"])

        # Estilo para os widgets
        style = ttk.Style()
        style.configure("Custom.TButton",
                        padding=10,
                        font=("Helvetica", 12),
                        background=COLORS["secondary"])

        # Frame principal dividido em duas partes
        main_frame = tk.Frame(root, bg=COLORS["background"], padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")

        # Criar frames para esquerda e direita
        left_frame = tk.Frame(main_frame, bg=COLORS["background"], width=600)
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))

        right_frame = tk.Frame(main_frame, bg=COLORS["background"], width=600)
        right_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=(10, 0))

        # Título para a seção de exercícios (lado esquerdo)
        title_frame_left = tk.Frame(left_frame, bg=COLORS["background"])
        title_frame_left.pack(pady=(0, 20))

        title_label_left = tk.Label(title_frame_left,
                            text="Exercícios Disponíveis",
                            font=("Helvetica", 24, "bold"),
                            fg=COLORS["primary"],
                            bg=COLORS["background"])
        title_label_left.pack()

        # Título para a seção de planos de treino (lado direito)
        title_frame_right = tk.Frame(right_frame, bg=COLORS["background"])
        title_frame_right.pack(pady=(0, 20))

        title_label_right = tk.Label(title_frame_right,
                            text="Planos de Treino",
                            font=("Helvetica", 24, "bold"),
                            fg=COLORS["primary"],
                            bg=COLORS["background"])
        title_label_right.pack()

        # Frame para os botões de exercícios (lado esquerdo)
        exercise_buttons_frame = tk.Frame(left_frame, bg=COLORS["background"])
        exercise_buttons_frame.pack(pady=20)

        # Grid para organizar os botões de exercícios
        buttons_grid = tk.Frame(exercise_buttons_frame, bg=COLORS["background"])
        buttons_grid.pack()

        # Função para iniciar exercício com tela dividida
        def start_exercise(exercise_name):
            try:
                # Armazenar referência da janela principal
                tk.Tk.root = root
                root.withdraw()
                run_split_screen(exercise_name)
                root.deiconify()
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao iniciar exercício: {str(e)}")
                root.deiconify()

        # Adicionar botões para cada exercício em uma grade de 2x3
        exercises = [
            "Agachamento", "Flexao", "Prancha",
            "Lunge", "Jumping Jacks", "Shoulder Press"
        ]

        for i, exercise in enumerate(exercises):
            row = i // 3
            col = i % 3
            exercise_frame = tk.Frame(buttons_grid, bg=COLORS["background"])
            exercise_frame.grid(row=row, column=col, padx=15, pady=15)
            btn = create_button(
                exercise_frame,
                exercise,
                lambda ex=exercise: start_exercise(ex)
            )
            btn.pack()

        # Frame para os planos de treino (lado direito)
        workout_plans_frame = tk.Frame(right_frame, bg=COLORS["background"])
        workout_plans_frame.pack(pady=20)

        # Lista de planos de treino
        plans_listbox = tk.Listbox(workout_plans_frame, 
                                font=("Helvetica", 12),
                                bg=COLORS["white"],
                                fg=COLORS["text"],
                                selectbackground=COLORS["secondary"],
                                height=10)
        plans_listbox.pack(fill="both", expand=True, pady=10)

        # Adicionar planos de treino à lista
        for plan_name in PLANOS_DISPONIVEIS.keys():
            plans_listbox.insert(tk.END, plan_name)

        # Frame para detalhes do plano selecionado
        plan_details_frame = tk.Frame(workout_plans_frame, bg=COLORS["background"])
        plan_details_frame.pack(fill="both", expand=True, pady=10)

        plan_details_text = tk.Text(plan_details_frame,
                                font=("Helvetica", 12),
                                bg=COLORS["white"],
                                fg=COLORS["text"],
                                height=10,
                                wrap=tk.WORD)
        plan_details_text.pack(fill="both", expand=True)

        def show_plan_details(event):
            try:
                selection = plans_listbox.curselection()
                if selection:
                    plan_name = plans_listbox.get(selection[0])
                    plan = PlanoTreino(plan_name, PLANOS_DISPONIVEIS[plan_name])
                    
                    # Limpar e atualizar detalhes
                    plan_details_text.delete(1.0, tk.END)
                    plan_details_text.insert(tk.END, f"Plano: {plan_name}\n\n")
                    plan_details_text.insert(tk.END, "Exercícios:\n")
                    for i, exercise in enumerate(plan.exercicios, 1):
                        plan_details_text.insert(tk.END, f"{i}. {exercise}\n")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao mostrar detalhes do plano: {str(e)}")

        # Vincular evento de seleção à função
        plans_listbox.bind('<<ListboxSelect>>', show_plan_details)

        # Botão para iniciar plano de treino selecionado
        def start_selected_plan():
            try:
                selection = plans_listbox.curselection()
                if selection:
                    plan_name = plans_listbox.get(selection[0])
                    plan = PlanoTreino(plan_name, PLANOS_DISPONIVEIS[plan_name])
                    # Armazenar referência da janela principal
                    tk.Tk.root = root
                    root.withdraw()
                    run_split_screen(plan.exercicio_atual(), plan)
                    root.deiconify()
                else:
                    messagebox.showwarning("Aviso", "Por favor, selecione um plano de treino.")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao iniciar plano: {str(e)}")
                root.deiconify()

        start_plan_button = create_button(workout_plans_frame,
                                        "Iniciar Plano Selecionado",
                                        start_selected_plan)
        start_plan_button.pack(pady=10)

        # Footer
        footer_frame = tk.Frame(main_frame, bg=COLORS["background"])
        footer_frame.pack(side=tk.BOTTOM, pady=20)

        tk.Label(footer_frame,
                text="Sistema de Análise de Exercícios © 2025",
                font=("Helvetica", 10),
                fg=COLORS["text"],
                bg=COLORS["background"]).pack()

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
