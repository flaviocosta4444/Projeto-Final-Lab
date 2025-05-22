import cv2
import numpy as np
import time
from pose_analysis import get_reference_pose, create_stick_figure, pontuacao, etapa_atual, exercicio_atual, set_exercicio, avaliar_angulo, calculate_angle
import mediapipe as mp
import glob
import os
import tkinter as tk
from tkinter import messagebox

def load_exercise_frames(exercise, width, height):
    folder = os.path.join('Exercicios', exercise)
    frame_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
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

def run_split_screen(exercicio, plano_treino=None):
    set_exercicio(exercicio)
    print(f"Iniciando tela dividida para: {exercicio}")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        return
    height, width, _ = frame.shape
    frames = load_exercise_frames(exercicio, width, height)
    num_frames = len(frames)
    frame_idx = 0
    frames_per_image = 7
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    exercicio_iniciado = False
    exercicio_concluido = False
    tempo_inicio = None
    tempo_duracao = 30

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            main_feedbacks = []

            if not exercicio_iniciado and results.pose_landmarks:
                exercicio_iniciado = True
                tempo_inicio = time.time()

            if exercicio_iniciado and not exercicio_concluido:
                tempo_atual = time.time()
                tempo_decorrido = tempo_atual - tempo_inicio
                if tempo_decorrido >= tempo_duracao:
                    exercicio_concluido = True
                    if plano_treino:
                        proximo_exercicio = plano_treino.proximo_exercicio()
                        if proximo_exercicio is not None:
                            cv2.putText(frame, "Exercício Concluído!", (width//2 - 150, height//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Próximo: {proximo_exercicio}", (width//2 - 150, height//2 + 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow("Exercício", frame)
                            cv2.waitKey(2000)
                            # Atualiza variáveis para o novo exercício, sem fechar a janela
                            exercicio = proximo_exercicio
                            set_exercicio(exercicio)
                            frames = load_exercise_frames(exercicio, width, height)
                            num_frames = len(frames)
                            frame_idx = 0
                            exercicio_iniciado = False
                            exercicio_concluido = False
                            tempo_inicio = None
                            continue
                        else:
                            cv2.putText(frame, "Plano de Treino Concluído!", (width//2 - 200, height//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "Retornando ao menu principal...", (width//2 - 200, height//2 + 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

            reference_angles = get_reference_pose()
            if num_frames > 0:
                model_frame = frames[(frame_idx // frames_per_image) % num_frames]
                frame_idx += 1
            else:
                model_frame = np.zeros((height, width, 3), dtype=np.uint8)

            mini_w = width // 4
            mini_h = height // 4
            mini_boneco = cv2.resize(model_frame, (mini_w, mini_h))
            x_offset = width - mini_w - 20
            y_offset = height - mini_h - 20
            frame[y_offset:y_offset+mini_h, x_offset:x_offset+mini_w] = mini_boneco

            if exercicio_iniciado and not exercicio_concluido:
                tempo_restante = max(0, tempo_duracao - (time.time() - tempo_inicio))
                cv2.putText(frame, f"Tempo restante: {int(tempo_restante)}s", (20, height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Sua pose", (50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Referência", (x_offset, y_offset - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Exercício", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Aplicação encerrada.")

if __name__ == "__main__":
    main()