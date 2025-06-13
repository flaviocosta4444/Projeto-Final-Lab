import cv2
import numpy as np
import time
from pose_analysis import get_reference_pose, create_stick_figure, pontuacao, etapa_atual, exercicio_atual, set_exercicio, avaliar_angulo, calculate_angle
import mediapipe as mp
import glob
import os
import tkinter as tk
from tkinter import messagebox

TURQUOISE = (61, 218, 215)  # Azul turquesa da página inicial

def load_exercise_frames(exercise, width, height):
    # Remove spaces from the exercise name to match folder names
    folder_name = exercise.replace(" ", "")
    folder = os.path.join('Exercicios', folder_name)
    frame_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    print(f"[DEBUG] Caminho da pasta: {folder}")
    print(f"[DEBUG] Imagens encontradas: {frame_paths}")

    frames = []
    for f in frame_paths:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        # Centralizar no fundo preto
        frame_bg = np.zeros((height, width, 3), dtype=np.uint8)
        y_offset = (height - new_h) // 2
        x_offset = (width - new_w) // 2
        frame_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        frames.append(frame_bg)
    return frames

def draw_rounded_box(img, top_left, bottom_right, color, radius=30, thickness=-1, alpha=1.0):
    overlay = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text_with_bg(img, text, org, font, font_scale, text_color, bg_color, thickness=2, pad_x=10, pad_y=10, radius=20, alpha=1.0):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    draw_rounded_box(img, (x, y - h - pad_y), (x + w + 2 * pad_x, y + pad_y), bg_color, radius, -1, alpha)
    cv2.putText(img, text, (x + pad_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def run_split_screen(exercicio, plano_treino=None):
    # Determine if it's a single exercise (string) or from a plan (tuple)
    if isinstance(exercicio, tuple):
        current_exercicio_name_raw = exercicio[0]
        current_exercicio_duration = exercicio[1]
    else:
        current_exercicio_name_raw = exercicio
        current_exercicio_duration = 30  # Default duration for single exercises (e.g., 30 seconds)

    # Remove accents from the exercise name for display and internal use
    current_exercicio_name = current_exercicio_name_raw.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ê", "e").replace("ô", "o").replace("â", "a").replace("õ", "o").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U").replace("Ç", "C").replace("Ã", "A").replace("Õ", "O")

    set_exercicio(current_exercicio_name)
    print(f"Iniciando tela dividida para: {current_exercicio_name}")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        return
    height, width, _ = frame.shape
    frames = load_exercise_frames(current_exercicio_name, width, height)
    num_frames = len(frames)
    frame_idx = 0
    frames_per_image = 7
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    exercicio_iniciado = False
    exercicio_concluido = False
    tempo_inicio = None
    tempo_duracao = current_exercicio_duration # Use the determined duration
    tempo_restante= tempo_duracao
    feedback_msg = ""
    feedback_color = (61, 218, 215)  # Azul esverdeado
    feedback_tick = True

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            main_feedbacks = []

            # If exercicio_iniciado is False, set it to True to start the timer (for testing without camera)
            if not exercicio_iniciado:
                exercicio_iniciado = True
                tempo_inicio = time.time()

            if exercicio_iniciado and not exercicio_concluido:
                tempo_atual = time.time()
                tempo_decorrido = tempo_atual - tempo_inicio
                if tempo_inicio is None:
                    tempo_inicio = tempo_atual
                    tempo_decorrido = 0
                else:
                    tempo_decorrido = tempo_atual - tempo_inicio
                if tempo_decorrido >= tempo_duracao:
                    exercicio_concluido = True
                    if plano_treino:
                        proximo_exercicio = plano_treino.proximo_exercicio()
                        if proximo_exercicio is not None:
                            # Style for "Exercício Concluído!" and "Próximo:"
                            box_margin_x = 100
                            box_width = width - 2 * box_margin_x
                            box_height = 150
                            box_y = height // 2 - box_height // 2

                            # Draw a larger rounded box for the notification
                            draw_rounded_box(frame,
                                             (box_margin_x, box_y),
                                             (box_margin_x + box_width, box_y + box_height),
                                             TURQUOISE, radius=40, alpha=0.95)

                            # Display "Exercício Concluído!"
                            put_text_with_bg(frame, "Exercicio Concluido!",
                                             (width // 2 - cv2.getTextSize("Exercicio Concluido!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 50),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

                            # Display "Próximo: [Exercise Name], [Duration] seg"
                            next_exercise_text_raw = f"Proximo: {proximo_exercicio[0]}, {proximo_exercicio[1]} seg"
                            next_exercise_text = next_exercise_text_raw.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ê", "e").replace("ô", "o").replace("â", "a").replace("õ", "o").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U").replace("Ç", "C").replace("Ã", "A").replace("Õ", "O")
                            put_text_with_bg(frame, next_exercise_text,
                                             (width // 2 - cv2.getTextSize(next_exercise_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 100),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

                            cv2.imshow("Exercício", frame)
                            cv2.waitKey(2000)
                            exercicio = proximo_exercicio
                            set_exercicio(exercicio)
                            frames = load_exercise_frames(exercicio, width, height)
                            num_frames = len(frames)
                            frame_idx = 0
                            exercicio_iniciado = False
                            exercicio_concluido = False
                            tempo_inicio = None
                            tempo_duracao = exercicio[1] # Set new duration from the next exercise
                            tempo_restante = tempo_duracao # Reset remaining time
                            continue
                        else:
                            # Style for "Plano de Treino Concluído!" and "Retornando ao menu principal..."
                            box_margin_x = 100
                            box_width = width - 2 * box_margin_x
                            box_height = 150
                            box_y = height // 2 - box_height // 2

                            # Draw a larger rounded box for the notification
                            draw_rounded_box(frame,
                                             (box_margin_x, box_y),
                                             (box_margin_x + box_width, box_y + box_height),
                                             TURQUOISE, radius=40, alpha=0.95)

                            put_text_with_bg(frame, "Plano de Treino Concluido!",
                                             (width // 2 - cv2.getTextSize("Plano de Treino Concluido!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 50),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)
                            put_text_with_bg(frame, "Retornando ao menu principal...",
                                             (width // 2 - cv2.getTextSize("Retornando ao menu principal...", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 100),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

                            cv2.imshow("Exercício", frame)
                            cv2.waitKey(2000)
                            cap.release()
                            cv2.destroyAllWindows()
                            plano_treino.reiniciar()
                            if hasattr(tk.Tk, 'root'):
                                tk.Tk.root.deiconify()
                            return

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                joints = {
                    "Quadril": ([lm[23].x, lm[23].y], [lm[24].x, lm[24].y], [lm[25].x, lm[25].y]),
                    "Tronco": ([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[24].x, lm[24].y]),
                    "Ombro Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                    "Ombro Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]),
                    "Cotovelo Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                    "Cotovelo Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
                }
                reference_angles = get_reference_pose()
                for name, (a, b, c) in joints.items():
                    a = [a[0] * width, a[1] * height]
                    b = [b[0] * width, b[1] * height]
                    c = [c[0] * width, c[1] * height]
                    angle = calculate_angle(a, b, c)
                    reference_angle = reference_angles.get(name, 0)
                    feedback, _ = avaliar_angulo(angle, reference_angle, name, exercicio, etapa_atual)
                    if feedback:
                        main_feedbacks.append(feedback)
                # Exemplo: feedback visual
                if main_feedbacks:
                    feedback_msg = main_feedbacks[0]
                    feedback_color = (61, 218, 215)
                    feedback_tick = True if "excelente" in feedback_msg.lower() else False
                else:
                    feedback_msg = ""
                    feedback_tick = False

            # --- OVERLAYS ---
            # Caixa topo esquerdo: Nome do exercício
            box_w = 420
            box_h = 90
            draw_rounded_box(frame, (0, 0), (box_w, box_h), TURQUOISE, radius=40, alpha=0.95)
            put_text_with_bg(frame, "Exercicio Atual:", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)
            put_text_with_bg(frame, current_exercicio_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), TURQUOISE, thickness=3, pad_x=0, pad_y=0, radius=0, alpha=0)

            # Caixa topo direito: Feedback
            if feedback_msg:
                # Ajustar largura do retângulo de feedback conforme o tamanho do texto
                max_fb_w = min(600, width - 60)  # nunca maior que a tela
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                feedback_msg_to_display = feedback_msg.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ê", "e").replace("ô", "o").replace("â", "a").replace("õ", "o").replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U").replace("Ç", "C").replace("Ã", "A").replace("Õ", "O")
                # Quebrar o texto em múltiplas linhas se necessário
                words = feedback_msg_to_display.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                    if w > max_fb_w - 100 and current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                if current_line:
                    lines.append(current_line)
                fb_h = 40 + 40 * len(lines)
                fb_w = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines] + [200]) + 100
                fb_w = min(fb_w, max_fb_w)
                x1 = width - fb_w - 30
                y1 = 30
                x2 = width - 30
                y2 = 30 + fb_h
                draw_rounded_box(frame, (x1, y1), (x2, y2), TURQUOISE, radius=30, alpha=0.95)
                if feedback_tick:
                    # Desenhar círculo verde com check
                    cv2.circle(frame, (x1+40, y1+30), 20, (57, 255, 128), -1)
                    put_text_with_bg(frame, "✓", (x1+28, y1+42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), TURQUOISE, thickness=3, pad_x=0, pad_y=0, radius=0, alpha=0)
                # Escrever cada linha do feedback
                for i, line in enumerate(lines):
                    put_text_with_bg(frame, line, (x1+80, y1+40 + i*40), font, font_scale, (255,255,255), TURQUOISE, thickness=thickness, pad_x=0, pad_y=0, radius=0, alpha=0)

            # Caixa canto inferior esquerdo: Modelo de referência
            mini_w = width // 5
            mini_h = height // 5
            model_x = 30
            # Puxar mais para cima (ajuste manual)
            model_y = height - mini_h - 80
            if model_y < 20:
                model_y = 20
            # Usar o frame de referência real
            if num_frames > 0:
                model_frame = frames[(frame_idx // frames_per_image) % num_frames]
                frame_idx += 1
            else:
                model_frame = np.zeros((mini_h, mini_w, 3), dtype=np.uint8)
            # Redimensionar e centralizar na caixa preta
            model_resized = cv2.resize(model_frame, (mini_w, mini_h))
            ref_box = np.zeros((mini_h+10, mini_w+10, 3), dtype=np.uint8)
            cv2.rectangle(ref_box, (0,0), (mini_w+9, mini_h+9), TURQUOISE, 4)
            ref_box[5:mini_h+5, 5:mini_w+5] = model_resized
            frame[model_y:model_y+mini_h+10, model_x:model_x+mini_w+10] = ref_box
            # Ajustar o texto para ficar logo abaixo da caixa
            texto_y = model_y + mini_h + 35
            if texto_y > height - 10:
                texto_y = height - 10
            put_text_with_bg(frame, "Modelo de Referencia", (model_x+10, texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

            # Caixa canto inferior direito: Tempo restante (apenas se estiver em plano de treino)
            if plano_treino is not None:
                tbox_w = 350
                tbox_h = 60
                draw_rounded_box(frame, (width-tbox_w-30, height-tbox_h-30), (width-30, height-30), TURQUOISE, radius=30, alpha=0.95)
                
                if exercicio_iniciado and not exercicio_concluido:
                    tempo_restante = max(0, tempo_duracao - int(time.time() - tempo_inicio))
                
                put_text_with_bg(frame, f"Tempo restante: {tempo_restante} seg", (width-tbox_w, height-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

            cv2.imshow("Exercício", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Aplicação encerrada.")

if __name__ == "__main__":
    main()