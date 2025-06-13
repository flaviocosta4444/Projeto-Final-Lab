import cv2
import numpy as np
import time
from pose_analysis import create_stick_figure, pontuacao, etapa_atual, exercicio_atual, set_exercicio, avaliar_angulo, calculate_angle, get_visible_side, calcular_pontuacao, extract_angles_from_landmarks
import mediapipe as mp
import glob
import os
import tkinter as tk
from tkinter import messagebox
import unicodedata

TURQUOISE = (61, 218, 215)  # Azul turquesa da página inicial

# Articulações críticas para verificação de confiança
# Baseado nos índices do MediaPipe Pose para landmarks
CRITICAL_LANDMARKS = [
    mp.solutions.pose.PoseLandmark.NOSE,
    mp.solutions.pose.PoseLandmark.LEFT_EYE,
    mp.solutions.pose.PoseLandmark.RIGHT_EYE,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    mp.solutions.pose.PoseLandmark.LEFT_HEEL,
    mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
    mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST
]

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

def clean_text_for_display(text):
    """
    Remove acentos e caracteres especiais de uma string para exibição no OpenCV.
    """
    text = str(text) # Garante que a entrada seja uma string
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.replace("?", "o") 
    return text

def run_split_screen(exercicio, plano_treino=None):
    # Determine if it's a single exercise (string) or from a plan (tuple)
    if isinstance(exercicio, tuple):
        current_exercicio_name_raw = exercicio[0]
        current_exercicio_duration = exercicio[1]
    else:
        current_exercicio_name_raw = exercicio
        current_exercicio_duration = 30  # Default duration for single exercises

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
    frames_per_image = 30 # Número de frames da câmera por imagem de referência
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Pré-processar todos os frames de referência para extrair os ângulos
    reference_angles_per_frame = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_ref:
        for ref_frame_img in frames:
            results_ref = pose_ref.process(cv2.cvtColor(ref_frame_img, cv2.COLOR_BGR2RGB))
            if results_ref.pose_landmarks:
                angles_ref = extract_angles_from_landmarks(results_ref.pose_landmarks, width, height)
                reference_angles_per_frame.append(angles_ref)
            else:
                reference_angles_per_frame.append({}) # Adiciona um dicionário vazio se não detectar pose

    exercicio_iniciado = False
    exercicio_concluido = False
    tempo_inicio = None
    tempo_duracao = current_exercicio_duration
    tempo_restante = tempo_duracao
    
    # Sistema de feedback em tempo real
    feedback_system = {
        'current_feedback': [],
        'feedback_history': [],
        'score': 0,
        'frame_count': 0,
        'last_feedback_time': time.time(),
        'feedback_cooldown': 2.0,  # Tempo mínimo entre feedbacks (em segundos)
        'min_confidence': 0.6,     # Confiança mínima para considerar uma detecção válida
        'feedback_threshold': 3     # Número mínimo de frames com o mesmo feedback para exibi-lo
    }

    def update_feedback_system(frame_feedbacks, landmarks):
        current_time = time.time()
        
        # Verificar confiança da detecção apenas para landmarks críticas do lado visível
        if landmarks:
            visible_side = get_visible_side(landmarks) # Determine o lado visível
            critical_visibility = []

            for lm_idx_enum in CRITICAL_LANDMARKS:
                lm_idx = lm_idx_enum.value
                # Se o lado não for 'both', ignore landmarks do lado oculto
                if visible_side == "left" and "RIGHT" in lm_idx_enum.name:
                    continue
                if visible_side == "right" and "LEFT" in lm_idx_enum.name:
                    continue

                if lm_idx < len(landmarks.landmark):
                    critical_visibility.append(landmarks.landmark[lm_idx].visibility)
            
            if critical_visibility:
                # Usar a média das visibilidades das landmarks críticas visíveis
                confidence = sum(critical_visibility) / len(critical_visibility)
            else:
                confidence = 0.0 # Sem landmarks críticas detectadas no lado visível

            if confidence < feedback_system['min_confidence']:
                return clean_text_for_display("Posicao nao detectada claramente. Ajuste sua posicao.")
        
        # Atualizar histórico de feedback
        if frame_feedbacks:
            feedback_system['feedback_history'].append(frame_feedbacks[0])
            # Manter apenas os últimos 30 feedbacks
            feedback_system['feedback_history'] = feedback_system['feedback_history'][-30:]
        
        # Verificar se passou tempo suficiente desde o último feedback
        if current_time - feedback_system['last_feedback_time'] < feedback_system['feedback_cooldown']:
            return None
        
        # Contar ocorrências do feedback mais recente
        if feedback_system['feedback_history']:
            recent_feedback = feedback_system['feedback_history'][-1]
            count = feedback_system['feedback_history'].count(recent_feedback)
            
            # Se o feedback se repetiu várias vezes, exibi-lo
            if count >= feedback_system['feedback_threshold']:
                feedback_system['last_feedback_time'] = current_time
                return recent_feedback
        
        return None

    def draw_feedback_box(frame, feedback, position, color):
        if not feedback:
            return
        
        # Configurar o texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        # Calcular tamanho do texto
        (text_width, text_height), _ = cv2.getTextSize(feedback, font, font_scale, thickness)
        
        # Calcular posição da caixa
        x, y = position
        box_width = text_width + 2 * padding
        box_height = text_height + 2 * padding
        
        # Desenhar caixa com cantos arredondados
        draw_rounded_box(frame, 
                        (x, y - box_height), 
                        (x + box_width, y), 
                        color, 
                        radius=20, 
                        alpha=0.8)
        
        # Desenhar texto
        cv2.putText(frame, 
                   feedback, 
                   (x + padding, y - padding), 
                   font, 
                   font_scale, 
                   (255, 255, 255), 
                   thickness, 
                   cv2.LINE_AA)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            main_feedbacks = []
            current_feedback = None # Inicializar current_feedback

            # Sincronizar etapa_atual com o frame de referência
            import pose_analysis
            pose_analysis.etapa_atual = frame_idx

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
                            put_text_with_bg(frame, clean_text_for_display("Exercicio Concluido!"),
                                             (width // 2 - cv2.getTextSize(clean_text_for_display("Exercicio Concluido!"), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 50),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

                            # Display "Próximo: [Exercise Name], [Duration] seg"
                            next_exercise_text_raw = f"Proximo: {proximo_exercicio[0]}, {proximo_exercicio[1]} seg"
                            put_text_with_bg(frame, clean_text_for_display(next_exercise_text_raw),
                                             (width // 2 - cv2.getTextSize(clean_text_for_display(next_exercise_text_raw), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 100),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

                            cv2.imshow("Exercício", frame)
                            cv2.waitKey(2000)
                            exercicio = proximo_exercicio
                            set_exercicio(exercicio)
                            frames = load_exercise_frames(exercicio[0], width, height) # Pass only the name
                            num_frames = len(frames)
                            frame_idx = 0
                            exercicio_iniciado = False
                            exercicio_concluido = False
                            tempo_inicio = None
                            tempo_duracao = exercicio[1] # Set new duration from the next exercise
                            tempo_restante = tempo_duracao # Reset remaining time
                            # Recalcular reference_angles_per_frame para o próximo exercício
                            reference_angles_per_frame = []
                            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_ref:
                                for ref_frame_img in frames:
                                    results_ref = pose_ref.process(cv2.cvtColor(ref_frame_img, cv2.COLOR_BGR2RGB))
                                    if results_ref.pose_landmarks:
                                        angles_ref = extract_angles_from_landmarks(results_ref.pose_landmarks, width, height)
                                        reference_angles_per_frame.append(angles_ref)
                                    else:
                                        reference_angles_per_frame.append({}) # Adiciona um dicionário vazio
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

                            put_text_with_bg(frame, clean_text_for_display("Plano de Treino Concluido!"),
                                             (width // 2 - cv2.getTextSize(clean_text_for_display("Plano de Treino Concluido!"), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 50),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)
                            put_text_with_bg(frame, clean_text_for_display("Retornando ao menu principal..."),
                                             (width // 2 - cv2.getTextSize(clean_text_for_display("Retornando ao menu principal..."), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2, box_y + 100),
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
                    "Quadril": ([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                                [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y], 
                                [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
                    "Tronco": ([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                               [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                               [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                    "Ombro Direito": ([lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y], 
                                      [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], 
                                      [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                    "Ombro Esquerdo": ([lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y], 
                                       [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                                       [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                    "Cotovelo Direito": ([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], 
                                         [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y], 
                                         [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                    "Cotovelo Esquerdo": ([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                                          [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y], 
                                          [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                    "Joelho Direito": ([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y], 
                                       [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y], 
                                       [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
                    "Joelho Esquerdo": ([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                                         [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y], 
                                         [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
                }

                # Agora, o reference_angles virá da lista pré-processada
                if frame_idx // frames_per_image < len(reference_angles_per_frame):
                    reference_angles = reference_angles_per_frame[frame_idx // frames_per_image]
                else:
                    reference_angles = {} # Se estiver fora do limite, use um dicionário vazio

                frame_feedbacks = []

                # Determinar o lado visível
                visible_side = get_visible_side(results.pose_landmarks)

                # Calcular todos os ângulos primeiro
                calculated_angles = {}
                for name, (a, b, c) in joints.items():
                    a_coord = [a[0] * width, a[1] * height]
                    b_coord = [b[0] * width, b[1] * height]
                    c_coord = [c[0] * width, c[1] * height]
                    calculated_angles[name] = calculate_angle(a_coord, b_coord, c_coord)

                # Aplicar a lógica de copiar ângulos do lado visível para o lado oculto
                if visible_side == "left":
                    if "Joelho Esquerdo" in calculated_angles and "Joelho Direito" in calculated_angles:
                        calculated_angles["Joelho Direito"] = calculated_angles["Joelho Esquerdo"]
                    if "Quadril Esquerdo" in calculated_angles and "Quadril Direito" in calculated_angles:
                        calculated_angles["Quadril Direito"] = calculated_angles["Quadril Esquerdo"]
                    if "Ombro Esquerdo" in calculated_angles and "Ombro Direito" in calculated_angles:
                        calculated_angles["Ombro Direito"] = calculated_angles["Ombro Esquerdo"]
                    if "Cotovelo Esquerdo" in calculated_angles and "Cotovelo Direito" in calculated_angles:
                        calculated_angles["Cotovelo Direito"] = calculated_angles["Cotovelo Esquerdo"]
                elif visible_side == "right":
                    if "Joelho Direito" in calculated_angles and "Joelho Esquerdo" in calculated_angles:
                        calculated_angles["Joelho Esquerdo"] = calculated_angles["Joelho Direito"]
                    if "Quadril Direito" in calculated_angles and "Quadril Esquerdo" in calculated_angles:
                        calculated_angles["Quadril Esquerdo"] = calculated_angles["Quadril Direito"]
                    if "Ombro Direito" in calculated_angles and "Ombro Esquerdo" in calculated_angles:
                        calculated_angles["Ombro Esquerdo"] = calculated_angles["Ombro Direito"]
                    if "Cotovelo Direito" in calculated_angles and "Cotovelo Esquerdo" in calculated_angles:
                        calculated_angles["Cotovelo Esquerdo"] = calculated_angles["Cotovelo Direito"]

                # Analisar cada articulação com os ângulos ajustados
                for name, angle in calculated_angles.items():
                    # Usar o reference_angle do frame atual do modelo de referência
                    ref_angle_for_joint = reference_angles.get(name, None)
                    if ref_angle_for_joint is not None:
                        feedback, _ = avaliar_angulo(angle, ref_angle_for_joint, name, exercicio, etapa_atual, landmarks=results.pose_landmarks)
                    if feedback:
                            frame_feedbacks.append(feedback)

                # Calcular a pontuação do frame usando a função calcular_pontuacao
                frame_score_calculated = calcular_pontuacao(calculated_angles, reference_angles)

                # Atualizar pontuação total (acumulando a pontuação calculada por frame)
                feedback_system['score'] += frame_score_calculated

                # Atualizar sistema de feedback
                current_feedback = update_feedback_system(frame_feedbacks, results.pose_landmarks)

                # Desenhar feedback em tempo real
                if current_feedback:
                    draw_feedback_box(frame,
                                      clean_text_for_display(current_feedback),
                                      (width - 400, 100),
                                      TURQUOISE)

                # Mostrar pontuação atual
                if feedback_system['frame_count'] > 0:
                    avg_score = feedback_system['score'] / feedback_system['frame_count']
                    score_text = f"Pontuacao: {avg_score:.2f}"
                    draw_feedback_box(frame,
                                      clean_text_for_display(score_text),
                                      (width - 200, 50),
                                      TURQUOISE)

                feedback_system['frame_count'] += 1

            # --- OVERLAYS ---
            # Caixa topo esquerdo: Nome do exercício
            box_w = 420
            box_h = 90
            draw_rounded_box(frame, (0, 0), (box_w, box_h), TURQUOISE, radius=40, alpha=0.95)
            put_text_with_bg(frame, clean_text_for_display("Exercicio Atual:"), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)
            put_text_with_bg(frame, clean_text_for_display(current_exercicio_name), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), TURQUOISE, thickness=3, pad_x=0, pad_y=0, radius=0, alpha=0)

            # Caixa topo direito: Feedback
            if current_feedback:
                # Ajustar largura do retângulo de feedback conforme o tamanho do texto
                max_fb_w = min(600, width - 60)  # nunca maior que a tela
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                feedback_msg_to_display = clean_text_for_display(current_feedback)
                # Quebrar o texto em múltiplas linhas se necessário
                words = feedback_msg_to_display.split()
                lines = []
                current_line_text = ""
                for word in words:
                    test_line = current_line_text + (" " if current_line_text else "") + word
                    (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                    if w > max_fb_w - 100 and current_line_text:
                        lines.append(current_line_text)
                        current_line_text = word
                    else:
                        current_line_text = test_line
                if current_line_text:
                    lines.append(current_line_text)
                fb_h = 40 + 40 * len(lines)
                fb_w = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines] + [200]) + 100
                fb_w = min(fb_w, max_fb_w)
                x1 = width - fb_w - 30
                y1 = 30
                x2 = width - 30
                y2 = 30 + fb_h
                draw_rounded_box(frame, (x1, y1), (x2, y2), TURQUOISE, radius=30, alpha=0.95)
                # Escrever cada linha do feedback
                for i, line in enumerate(lines):
                    put_text_with_bg(frame, line, (x1 + 80, y1 + 40 + i * 40), font, font_scale, (255, 255, 255), TURQUOISE, thickness=thickness, pad_x=0, pad_y=0, radius=0, alpha=0)

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
            ref_box = np.zeros((mini_h + 10, mini_w + 10, 3), dtype=np.uint8)
            cv2.rectangle(ref_box, (0, 0), (mini_w + 9, mini_h + 9), TURQUOISE, 4)
            ref_box[5:mini_h + 5, 5:mini_w + 5] = model_resized
            frame[model_y:model_y + mini_h + 10, model_x:model_x + mini_w + 10] = ref_box
            # Ajustar o texto para ficar logo abaixo da caixa
            texto_y = model_y + mini_h + 35
            if texto_y > height - 10:
                texto_y = height - 10
            put_text_with_bg(frame, clean_text_for_display("Modelo de Referencia"), (model_x + 10, texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

            # Caixa canto inferior direito: Tempo restante (apenas se estiver em plano de treino)
            if plano_treino is not None:
                tbox_w = 350
                tbox_h = 60
                draw_rounded_box(frame, (width - tbox_w - 30, height - tbox_h - 30), (width - 30, height - 30), TURQUOISE, radius=30, alpha=0.95)
                
                if exercicio_iniciado and not exercicio_concluido:
                    tempo_restante = max(0, tempo_duracao - int(time.time() - tempo_inicio))
                
                put_text_with_bg(frame, clean_text_for_display(f"Tempo restante: {tempo_restante} seg"), (width - tbox_w, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), TURQUOISE, thickness=2, pad_x=0, pad_y=0, radius=0, alpha=0)

            cv2.imshow("Exercício", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Aplicacao encerrada.")

if __name__ == "__main__":
    main()