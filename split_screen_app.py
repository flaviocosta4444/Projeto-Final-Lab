import cv2
import numpy as np
import time
from pose_analysis import (
    get_reference_pose, create_stick_figure, pontuacao,
    etapa_atual, exercicio_atual, set_exercicio, avaliar_angulo,
    calculate_angle, atualizar_etapa, calcular_pontuacao
)
import mediapipe as mp
import glob
import os

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

def run_split_screen(exercicio):
    set_exercicio(exercicio)
    print(f"Iniciando tela dividida para: {exercicio}")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        return
    height, width, _ = frame.shape
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    ultimo_tempo = time.time()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Atualizar a etapa com base no tempo
            atualizar_etapa()

            # Obter ângulos esperados da etapa atual
            reference_angles = get_reference_pose()

            # Processar pose da pessoa
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            user_angles = {}
            feedbacks = []

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                joints = {
                    "Quadril": ([lm[23].x, lm[23].y], [lm[24].x, lm[24].y], [lm[25].x, lm[25].y]),
                    "Tronco": ([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[24].x, lm[24].y]),
                    "Ombro Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                    "Ombro Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]),
                    "Cotovelo Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                    "Cotovelo Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]),
                    "Joelho Direito": ([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]),
                    "Joelho Esquerdo": ([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
                }

                for name, (a, b, c) in joints.items():
                    a = [a[0] * width, a[1] * height]
                    b = [b[0] * width, b[1] * height]
                    c = [c[0] * width, c[1] * height]
                    angle = calculate_angle(a, b, c)
                    user_angles[name] = angle

                    ref_angle = reference_angles.get(name, None)
                    if ref_angle is not None:
                        feedback, _ = avaliar_angulo(angle, ref_angle, name, exercicio, etapa_atual)
                        if feedback:
                            feedbacks.append(feedback)
                        cv2.putText(frame, f"{name}: {int(angle)}°", (10, 30 + len(feedbacks)*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calcular pontuação sincronizada
                global pontuacao
                pontuacao = calcular_pontuacao(user_angles, reference_angles)

            # Mostrar feedback
            for i, fb in enumerate(feedbacks[:3]):
                cv2.putText(frame, fb, (10, height - 90 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Mostrar pontuação
            cv2.putText(frame, f"Pontuação: {pontuacao}%", (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar etapa
            cv2.putText(frame, f"Etapa: {etapa_atual + 1}", (width - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

            cv2.imshow("Siga o Modelo em Tempo Real", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 