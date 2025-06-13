import cv2
import mediapipe as mp
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_3d import Model3D  # Import the 3D model class
import os
import tkinter as tk
import unicodedata # Adicionar esta importação

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Variáveis globais
exercicio_atual = "Agachamento"
etapa_atual = 0
ultimo_tempo_etapa = 0
pontuacao = 0
TEMPO_POR_ETAPA = 3  # segundos por etapa do exercício - aumentado para 3 segundos

# Inicializar modelo 3D
model_3d = Model3D()

TURQUOISE = (61, 218, 215)  # Azul turquesa da página inicial

# Articulações críticas para verificação de confiança
# Baseado nos índices do MediaPipe Pose para landmarks
CRITICAL_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST
]

def clean_text_for_display(text):
    """
    Remove acentos e caracteres especiais de uma string para exibição no OpenCV.
    """
    text = str(text) # Garante que a entrada seja uma string
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Substituir '?' por 'o' para palavras como 'Posicao' (caso o unicodedata não resolva)
    text = text.replace("?", "o") 
    return text

def set_exercicio(nome):
    global exercicio_atual
    exercicio_atual = nome

def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return round(ang, 2)

def extract_angles_from_landmarks(landmarks, width, height):
    """
    Extrai e calcula os ângulos de articulação a partir dos landmarks do MediaPipe.
    """
    lm = landmarks.landmark
    joints_data = {
        "Quadril": ([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                    [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                    [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]),
        "Tronco": ([lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y], 
                   [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                   [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
        "Ombro Direito": ([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y], 
                          [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], 
                          [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
        "Ombro Esquerdo": ([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                           [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                           [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
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
                            [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
    }
    
    calculated_angles = {}
    for name, (p1, p2, p3) in joints_data.items():
        try:
            # Converter para coordenadas de pixel
            p1_coord = [p1[0] * width, p1[1] * height]
            p2_coord = [p2[0] * width, p2[1] * height]
            p3_coord = [p3[0] * width, p3[1] * height]
            angle = calculate_angle(p1_coord, p2_coord, p3_coord)
            calculated_angles[name] = angle
        except Exception as e:
            # print(f"Erro ao calcular ângulo para {name}: {e}")
            calculated_angles[name] = 0 # Valor padrão em caso de erro
            
    return calculated_angles

def draw_angle(image, point, angle, feedback, color=(0, 255, 0), y_offset=0):
    """
    Desenha o ângulo e o feedback na imagem.
    """
    h, w = image.shape[:2]
    x, y = int(point[0] * w), int(point[1] * h)
    
    # Desenhar o ângulo
    cv2.putText(
        image,
        clean_text_for_display(f"{int(angle)}o"), # Usar clean_text_for_display e 'o' para grau
        (x - 20, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    
    # Desenhar o feedback
    if feedback:
        cv2.putText(
            image,
            clean_text_for_display(feedback),
            (10, 30 + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

def get_visible_side(landmarks):
    left_confidence = 0
    right_confidence = 0

    # Calculate average visibility for key left-side landmarks
    left_landmarks_to_check = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]
    left_visibilities = [landmarks.landmark[lm.value].visibility for lm in left_landmarks_to_check if lm.value < len(landmarks.landmark)]
    if left_visibilities:
        left_confidence = sum(left_visibilities) / len(left_visibilities)

    # Calculate average visibility for key right-side landmarks
    right_landmarks_to_check = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    right_visibilities = [landmarks.landmark[lm.value].visibility for lm in right_landmarks_to_check if lm.value < len(landmarks.landmark)]
    if right_visibilities:
        right_confidence = sum(right_visibilities) / len(right_visibilities)

    if left_confidence > right_confidence + 0.1: # Small margin for clear dominance
        return "left"
    elif right_confidence > left_confidence + 0.1:
        return "right"
    else:
        return "both" # Neither side is clearly dominant or both are visible

def avaliar_angulo(angulo_atual, angulo_referencia, nome_articulacao, exercicio_atual, etapa_atual, landmarks=None):
    """
    Avalia se o ângulo atual está próximo do ângulo de referência.
    Retorna feedback.
    """
    # Margens de tolerância específicas para cada articulação e exercício
    margens = {
        "Agachamento": {
            "Joelho Direito": 15,
            "Joelho Esquerdo": 15,
            "Quadril": 15,
            "Tronco": 10,
            "Cotovelo Direito": 20,
            "Cotovelo Esquerdo": 20,
            "Ombro Direito": 20,
            "Ombro Esquerdo": 20
        },
        "Flexao": {
            "Cotovelo Direito": 10,
            "Cotovelo Esquerdo": 10,
            "Quadril": 15,
            "Tronco": 10,
            "Joelho Direito": 15,
            "Joelho Esquerdo": 15,
            "Ombro Direito": 15,
            "Ombro Esquerdo": 15
        },
        "Abdominais": { # Adicionando margens para Abdominais
            "Tronco": 10,
            "Quadril": 10,
            "Joelho Direito": 10,
            "Joelho Esquerdo": 10
        }
    }
    
    # Obter a margem de tolerância para a articulação atual
    margem = margens.get(exercicio_atual, {}).get(nome_articulacao, 15)
    
    feedback = ""
    color = (255, 255, 255) # Cor padrão branca

    # Adicionar lógica para ignorar feedback de lados ocultos
    if landmarks and ("Esquerdo" in nome_articulacao or "Direito" in nome_articulacao):
        visible_side = get_visible_side(landmarks)
        if visible_side == "left" and "Direito" in nome_articulacao:
            return "", (255, 255, 255) # Não dar feedback para o lado direito se o esquerdo for o visível
        elif visible_side == "right" and "Esquerdo" in nome_articulacao:
            return "", (255, 255, 255) # Não dar feedback para o lado esquerdo se o direito for o visível
    
    # Calcular a diferença absoluta
    diferenca = abs(angulo_atual - angulo_referencia)
    
    # Determinar cor com base na diferença
    if diferenca <= margem:
        color = (0, 255, 0)  # Verde: bom
    elif diferenca <= margem * 2:
        color = (255, 255, 0)  # Amarelo: atenção
    else:
        color = (255, 0, 0)  # Vermelho: incorreto
    
    # Gerar feedback específico para cada exercício e etapa
    if exercicio_atual == "Flexao":
        if nome_articulacao in ["Cotovelo Direito", "Cotovelo Esquerdo"]:
            if etapa_atual == 0:  # Posição inicial
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Estenda mais o braco"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Mantenha os bracos estendidos"
            else:  # Posição baixa
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Desca mais"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Mantenha a posicao baixa"
        elif nome_articulacao == "Tronco":
            if angulo_atual > angulo_referencia + margem:
                feedback = "Mantenha o corpo reto"
            elif angulo_atual < angulo_referencia - margem:
                feedback = "Nao arqueie as costas"
        elif nome_articulacao == "Quadril":
            if angulo_atual > angulo_referencia + margem:
                feedback = "Mantenha o quadril alinhado"
            elif angulo_atual < angulo_referencia - margem:
                feedback = "Nao deixe o quadril cair"
    
    elif exercicio_atual == "Agachamento":
        if nome_articulacao in ["Joelho Direito", "Joelho Esquerdo"]:
            if etapa_atual == 1: # Meio caminho
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Desca mais o quadril"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Suba um pouco"
            elif etapa_atual == 2: # Agachamento completo
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Aprofunde mais o agachamento"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Mantenha a posicao de agachamento"
        elif nome_articulacao == "Quadril":
            if etapa_atual == 1 or etapa_atual == 2:
                if angulo_atual < angulo_referencia - margem:
                    feedback = "Quadril: Desca mais"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = "Quadril: Nao suba demais"
        elif nome_articulacao == "Tronco":
            if angulo_atual < angulo_referencia - margem:
                feedback = "Tronco: Nao incline tanto para frente"
            elif angulo_atual > angulo_referencia + margem:
                feedback = "Tronco: Mantenha o peito aberto"

    elif exercicio_atual == "Abdominais":
        if nome_articulacao == "Tronco":
            if etapa_atual == 0: # Para abdominais, a etapa 0 é a posição inicial ou final
                # Assumindo que a referencia é para o ponto de maior contração
                if angulo_atual < angulo_referencia - margem:
                    feedback = "Tronco: Suba mais (contraia o abdomen)"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = "Tronco: Desca mais devagar"
        elif nome_articulacao in ["Joelho Direito", "Joelho Esquerdo"]:
            if etapa_atual == 0:
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Nao estenda demais a perna"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Flexione mais a perna"
    
    return feedback, color

def get_reference_pose():
    """
    Retorna os valores de ângulos de referência para a etapa atual do exercício selecionado.
    """
    global exercicio_atual, etapa_atual
    
    # Definições de ângulos de referência para cada exercício e etapa
    referencias = {
        "Agachamento": [
            {  # Etapa 0: Posição inicial - de pé
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Quadril": 175,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 0,
                "Ombro Esquerdo": 0
            },
            {  # Etapa 1: Meio caminho
                "Joelho Direito": 120,
                "Joelho Esquerdo": 120,
                "Quadril": 120,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 0,
                "Ombro Esquerdo": 0
            },
            {  # Etapa 2: Posição agachada completa
                "Joelho Direito": 90,
                "Joelho Esquerdo": 90,
                "Quadril": 90,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 0,
                "Ombro Esquerdo": 0
            }
        ],
        "Flexao": [
            {  # Etapa 0: Posição inicial
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Quadril": 170,
                "Tronco": 0,  # Horizontalizado
                "Joelho Direito": 170,
                "Joelho Esquerdo": 170,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            },
            {  # Etapa 1: Meio caminho
                "Cotovelo Direito": 120,
                "Cotovelo Esquerdo": 120,
                "Quadril": 170,
                "Tronco": 0,  # Horizontalizado
                "Joelho Direito": 170,
                "Joelho Esquerdo": 170,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            },
            {  # Etapa 2: Posição baixa
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Quadril": 170,
                "Tronco": 0,  # Horizontalizado
                "Joelho Direito": 170,
                "Joelho Esquerdo": 170,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            }
        ],
        "Prancha": [
            {  # Etapa 0: Manter a prancha
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Quadril": 170,
                "Tronco": 0,  # Horizontalizado
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            }
        ],
        "Lunge": [
            {  # Etapa 0: Posição inicial
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Quadril": 175,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 1: Posição de lunge direito
                "Joelho Direito": 90,
                "Joelho Esquerdo": 165,
                "Quadril": 120,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 2: Posição de lunge esquerdo
                "Joelho Direito": 165,
                "Joelho Esquerdo": 90,
                "Quadril": 120,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            }
        ],
        "Jumping Jacks": [
            {  # Etapa 0: Posição inicial
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Quadril": 175,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 1: Posição com braços e pernas abertos
                "Joelho Direito": 120,
                "Joelho Esquerdo": 120,
                "Quadril": 120,
                "Tronco": 90,
                "Cotovelo Direito": 175,
                "Cotovelo Esquerdo": 175,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            }
        ],
        "Abdominais": [
            {  # Etapa 0: Posição inicial (deitado)
                "Joelho Direito": 90,
                "Joelho Esquerdo": 90,
                "Quadril": 0,  # Deitado
                "Tronco": 0,   # Deitado
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 1: Posição elevada
                "Joelho Direito": 90,
                "Joelho Esquerdo": 90,
                "Quadril": 45,  # Semi-elevado
                "Tronco": 45,   # Semi-elevado
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            }
        ]
    }
    
    # Obter valores do exercício atual
    if exercicio_atual not in referencias:
        return {}
    
    # Garantir que etapa_atual é válida
    num_etapas = len(referencias[exercicio_atual])
    if etapa_atual >= num_etapas:
        etapa_atual = 0
    
    return referencias[exercicio_atual][etapa_atual]

def atualizar_etapa():
    """
    Atualiza a etapa atual do exercício com base no tempo decorrido
    """
    global etapa_atual, ultimo_tempo_etapa, exercicio_atual
    
    # Definir quantidade de etapas por exercício
    etapas_por_exercicio = {
        "Agachamento": 3,
        "Flexao": 3,
        "Prancha": 1,
        "Lunge": 3,
        "Jumping Jacks": 2,
        "Abdominais": 2
    }
    
    # Obter o número total de etapas para o exercício atual
    num_etapas = etapas_por_exercicio.get(exercicio_atual, 1)
    
    # Verificar se é hora de mudar para a próxima etapa
    tempo_atual = time.time()
    if tempo_atual - ultimo_tempo_etapa >= TEMPO_POR_ETAPA:
        etapa_atual = (etapa_atual + 1) % num_etapas
        ultimo_tempo_etapa = tempo_atual

def calcular_pontuacao(user_joints, reference_values):
    """
    Calcula a pontuação com base na proximidade dos ângulos do usuário
    em relação aos ângulos de referência da etapa atual.
    """
    if not user_joints or not reference_values:
        return 0.0
    
    total_diff = 0
    count = 0
    
    for joint_name, ref_angle in reference_values.items():
        if joint_name in user_joints:
            user_angle = user_joints[joint_name]
            # Calcular diferença absoluta entre ângulos
            diff = abs(user_angle - ref_angle)
            # Normalizar a diferença para um valor entre 0 e 1
            normalized_diff = min(diff / 180.0, 1.0)
            total_diff += normalized_diff
            count += 1
    
    if count == 0:
        return 0.0
    
    # Calcular pontuação média (1.0 - diferença média)
    average_diff = total_diff / count
    score = 1.0 - average_diff
    
    return score

def draw_reference_pose(image, reference_values, user_angles):
    """
    Desenha a pose de referência na imagem e mostra a pontuação
    """
    global pontuacao, etapa_atual
    
    if not reference_values:
        return
    
    # Converter imagem para processamento com PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Desenhar informações sobre a etapa e exercício
    draw.text((10, 10), f"Exercício: {exercicio_atual}", fill=(255, 255, 255), font=font)
    draw.text((10, 40), f"Etapa: {etapa_atual + 1}", fill=(255, 255, 255), font=font)
    
    # Desenhar pontuação
    cor_pontuacao = (0, 255, 0)  # Verde para pontuação boa
    if pontuacao < 70:
        cor_pontuacao = (255, 255, 0)  # Amarelo para pontuação média
    if pontuacao < 40:
        cor_pontuacao = (255, 0, 0)  # Vermelho para pontuação ruim
    
    draw.text((10, 70), f"Pontuação: {pontuacao}%", fill=cor_pontuacao, font=font)
    
    # Desenhar informações sobre os ângulos de referência
    y_pos = 110
    draw.text((10, y_pos), "Ângulos de Referência:", fill=(255, 255, 255), font=small_font)
    y_pos += 25
    
    for joint_name, ref_angle in reference_values.items():
        user_angle = user_angles.get(joint_name, None)
        
        if user_angle is not None:
            # Obter feedback granular e cor da função avaliar_angulo
            feedback, cor = avaliar_angulo(user_angle, ref_angle, joint_name, exercicio_atual, etapa_atual)
            
            if feedback: # Se houver feedback específico, use-o
                texto = feedback
            else: # Caso contrário, use a informação padrão do ângulo
                texto = f"{joint_name}: {ref_angle}° (você: {user_angle:.0f}°)"
            
        draw.text((10, y_pos), texto, fill=cor, font=small_font)
        y_pos += 20
    
    # Converter de volta para formato OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image[:,:,:] = image_bgr[:,:,:]

def analyze_pose(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, _ = image.shape
            lm = results.pose_landmarks.landmark

            joints = {
                "Cotovelo Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                "Cotovelo Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]),
                "Joelho Direito": ([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]),
                "Joelho Esquerdo": ([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]),
                "Quadril": ([lm[23].x, lm[23].y], [lm[24].x, lm[24].y], [lm[25].x, lm[25].y]),
                "Tronco": ([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[24].x, lm[24].y]),
                "Ombro Direito": ([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]),
                "Ombro Esquerdo": ([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
            }

            y_offset = 0
            
            # Armazenar ângulos calculados
            calculated_angles = {}
            
            for name, (a, b, c) in joints.items():
                a = [a[0] * w, a[1] * h]
                b = [b[0] * w, b[1] * h]
                c = [c[0] * w, c[1] * h]

                angle = calculate_angle(a, b, c)
                calculated_angles[name] = angle
                feedback, color = avaliar_angulo(angle, angle, name, exercicio_atual, etapa_atual)
                
                draw_angle(image, b, angle, feedback, color=color, y_offset=y_offset)
                y_offset += 30
            
            return image, calculated_angles
        
        return image, {}

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERRO] Não foi possível abrir a imagem: {image_path}")
        return
    image, _ = analyze_pose(image)
    cv2.imshow('Análise de Imagem', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_stick_figure(angles, width, height):
    """
    Cria uma imagem com um boneco de palito representando os ângulos especificados.
    Versão melhorada para representar movimentos de exercícios com maior precisão.
    """
    global exercicio_atual, etapa_atual
    
    # Criar uma imagem em branco
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Definir cores
    BG_COLOR = (50, 50, 50)  # Cinza escuro para o fundo
    STICK_COLOR = (0, 255, 0)  # Verde para o boneco
    JOINT_COLOR = (255, 255, 0)  # Amarelo para as articulações
    TEXT_COLOR = (255, 255, 255)  # Branco para o texto
    
    # Preencher o fundo
    img[:] = BG_COLOR
    
    # Centro da imagem
    center_x = width // 2
    center_y = height // 2
    
    # Escala do boneco (ajustada para melhor visualização)
    scale = min(width, height) // 3
    
    # Mostrar o nome do exercício e a etapa
    cv2.putText(img, f"{exercicio_atual} - Etapa {etapa_atual + 1}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    
    # Posição da cabeça ajustada para cada exercício
    head_y_offset = 0
    if exercicio_atual == "Agachamento":
        if etapa_atual == 1:  # Posição agachada
            head_y_offset = scale // 5
    elif exercicio_atual == "Flexao":
        # Na flexão, o corpo fica na horizontal
        head_y_offset = scale // 3
    elif exercicio_atual == "Prancha":
        # Prancha em posição horizontal
        head_y_offset = scale // 3
    
    # Posições de referência
    head_center = (center_x, center_y - scale // 3 + head_y_offset)
    
    # Desenhar cabeça
    cv2.circle(img, head_center, scale // 7, STICK_COLOR, -1)
    
    # Ângulos específicos para cada exercício
    quadril_angle = angles.get("Quadril", 170)
    torso_angle = angles.get("Tronco", 90)
    joelho_direito_angle = angles.get("Joelho Direito", 170)
    joelho_esquerdo_angle = angles.get("Joelho Esquerdo", 170)
    cotovelo_direito_angle = angles.get("Cotovelo Direito", 160)
    cotovelo_esquerdo_angle = angles.get("Cotovelo Esquerdo", 160)
    ombro_direito_angle = angles.get("Ombro Direito", 90)
    ombro_esquerdo_angle = angles.get("Ombro Esquerdo", 90)
    
    # Ajustar ângulos com base no exercício
    if exercicio_atual == "Flexao":
        # Na flexão, o tronco fica paralelo ao chão
        torso_angle = 0
        if etapa_atual == 0:  # Posição alta
            cotovelo_direito_angle = 160
            cotovelo_esquerdo_angle = 160
        else:  # Posição baixa
            cotovelo_direito_angle = 90
            cotovelo_esquerdo_angle = 90
    elif exercicio_atual == "Prancha":
        # Na prancha, o corpo fica em posição horizontal
        torso_angle = 0
        cotovelo_direito_angle = 90
        cotovelo_esquerdo_angle = 90
    elif exercicio_atual == "Deadlift":
        if etapa_atual == 1:  # Posição baixa
            torso_angle = 45  # Inclinação para frente
    elif exercicio_atual == "Shoulder Press":
        if etapa_atual == 0:  # Posição inicial
            cotovelo_direito_angle = 90
            cotovelo_esquerdo_angle = 90
        else:  # Posição final
            cotovelo_direito_angle = 170
            cotovelo_esquerdo_angle = 170
    
    # Tronco - ajustado para representar melhor o movimento
    torso_rad = math.radians(torso_angle - 90)  # Corrigir para apontar para baixo por padrão
    
    # Comprimento do tronco varia conforme o exercício
    torso_length = scale * 0.8
    if exercicio_atual == "Agachamento" and etapa_atual == 1:
        torso_length = scale * 0.7  # Tronco menor durante o agachamento
    
    torso_end_x = int(head_center[0] + torso_length * math.cos(torso_rad))
    torso_end_y = int(head_center[1] + torso_length * math.sin(torso_rad))
    
    # Desenhar tronco
    cv2.line(img, head_center, (torso_end_x, torso_end_y), STICK_COLOR, 4)
    cv2.circle(img, (torso_end_x, torso_end_y), 5, JOINT_COLOR, -1)  # Articulação do quadril
    
    # Pernas - ajustadas para representar melhor os exercícios
    leg_length = scale * 0.9
    thigh_length = leg_length * 0.5  # Comprimento da coxa
    calf_length = leg_length * 0.5   # Comprimento da panturrilha
    
    # Ajustar posição das pernas conforme o exercício
    hip_angle_offset = 0
    if exercicio_atual == "Agachamento":
        if etapa_atual == 1:  # Posição agachada
            hip_angle_offset = -30  # Abrir mais as pernas durante o agachamento
    elif exercicio_atual == "Lunge":
        if etapa_atual in [1, 2]:  # Posição de lunge
            hip_angle_offset = -20  # Abrir mais as pernas
    
    # Articulação do quadril - centro das pernas
    hip_joint = (torso_end_x, torso_end_y)
    
    # Perna direita
    right_hip_rad = torso_rad - math.radians(30 + hip_angle_offset)  # Ângulo do quadril direito
    right_knee_x = int(hip_joint[0] + thigh_length * math.cos(right_hip_rad))
    right_knee_y = int(hip_joint[1] + thigh_length * math.sin(right_hip_rad))
    
    # Ângulo do joelho direito com base no valor de referência
    knee_adjustment = math.radians(180 - joelho_direito_angle)
    right_knee_rad = right_hip_rad + knee_adjustment
    
    right_ankle_x = int(right_knee_x + calf_length * math.cos(right_knee_rad))
    right_ankle_y = int(right_knee_y + calf_length * math.sin(right_knee_rad))
    
    # Desenhar coxa direita
    cv2.line(img, hip_joint, (right_knee_x, right_knee_y), STICK_COLOR, 4)
    cv2.circle(img, (right_knee_x, right_knee_y), 5, JOINT_COLOR, -1)  # Articulação do joelho
    
    # Desenhar perna direita
    cv2.line(img, (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), STICK_COLOR, 4)
    cv2.circle(img, (right_ankle_x, right_ankle_y), 5, JOINT_COLOR, -1)  # Articulação do tornozelo
    
    # Perna esquerda
    left_hip_rad = torso_rad + math.radians(30 + hip_angle_offset)  # Ângulo do quadril esquerdo
    left_knee_x = int(hip_joint[0] + thigh_length * math.cos(left_hip_rad))
    left_knee_y = int(hip_joint[1] + thigh_length * math.sin(left_hip_rad))
    
    # Ângulo do joelho esquerdo com base no valor de referência
    knee_adjustment = math.radians(180 - joelho_esquerdo_angle)
    left_knee_rad = left_hip_rad - knee_adjustment
    
    left_ankle_x = int(left_knee_x + calf_length * math.cos(left_knee_rad))
    left_ankle_y = int(left_knee_y + calf_length * math.sin(left_knee_rad))
    
    # Desenhar coxa esquerda
    cv2.line(img, hip_joint, (left_knee_x, left_knee_y), STICK_COLOR, 4)
    cv2.circle(img, (left_knee_x, left_knee_y), 5, JOINT_COLOR, -1)  # Articulação do joelho
    
    # Desenhar perna esquerda
    cv2.line(img, (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), STICK_COLOR, 4)
    cv2.circle(img, (left_ankle_x, left_ankle_y), 5, JOINT_COLOR, -1)  # Articulação do tornozelo
    
    # Braços - ajustados para cada exercício
    arm_length = scale * 0.75
    upper_arm_length = arm_length * 0.5  # Comprimento do braço superior
    forearm_length = arm_length * 0.5    # Comprimento do antebraço
    
    # Ombros - ponto de conexão dos braços
    shoulder_offset_x = 0
    shoulder_offset_y = -scale // 10
    
    # Ajustar posição dos braços conforme o exercício
    if exercicio_atual == "Flexao" or exercicio_atual == "Prancha":
        shoulder_offset_y = 0
    elif exercicio_atual == "Shoulder Press":
        shoulder_offset_y = -scale // 5
    
    right_shoulder_x = int(head_center[0] + shoulder_offset_x)
    right_shoulder_y = int(head_center[1] + shoulder_offset_y)
    left_shoulder_x = int(head_center[0] - shoulder_offset_x)
    left_shoulder_y = int(head_center[1] + shoulder_offset_y)
    
    # Braço direito
    right_shoulder_rad = math.radians(ombro_direito_angle - 180)
    if exercicio_atual == "Shoulder Press":
        if etapa_atual == 0:
            right_shoulder_rad = math.radians(0)
        else:
            right_shoulder_rad = math.radians(90)
    elif exercicio_atual == "Flexao" or exercicio_atual == "Prancha":
        right_shoulder_rad = math.radians(180)  # Braço para baixo na flexão/prancha
    
    right_elbow_x = int(right_shoulder_x + upper_arm_length * math.cos(right_shoulder_rad))
    right_elbow_y = int(right_shoulder_y + upper_arm_length * math.sin(right_shoulder_rad))
    
    # Ângulo do cotovelo direito
    elbow_adjustment = math.radians(180 - cotovelo_direito_angle)
    right_elbow_rad = right_shoulder_rad + elbow_adjustment
    
    right_wrist_x = int(right_elbow_x + forearm_length * math.cos(right_elbow_rad))
    right_wrist_y = int(right_elbow_y + forearm_length * math.sin(right_elbow_rad))
    
    # Desenhar braço direito
    cv2.line(img, (right_shoulder_x, right_shoulder_y), (right_elbow_x, right_elbow_y), STICK_COLOR, 4)
    cv2.circle(img, (right_elbow_x, right_elbow_y), 5, JOINT_COLOR, -1)  # Articulação do cotovelo
    
    # Desenhar antebraço direito
    cv2.line(img, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), STICK_COLOR, 4)
    cv2.circle(img, (right_wrist_x, right_wrist_y), 5, JOINT_COLOR, -1)  # Articulação do pulso
    
    # Braço esquerdo
    left_shoulder_rad = math.radians(ombro_esquerdo_angle - 180)
    if exercicio_atual == "Shoulder Press":
        if etapa_atual == 0:
            left_shoulder_rad = math.radians(180)
        else:
            left_shoulder_rad = math.radians(90)
    elif exercicio_atual == "Flexao" or exercicio_atual == "Prancha":
        left_shoulder_rad = math.radians(0)  # Braço para baixo na flexão/prancha
    
    left_elbow_x = int(left_shoulder_x + upper_arm_length * math.cos(left_shoulder_rad))
    left_elbow_y = int(left_shoulder_y + upper_arm_length * math.sin(left_shoulder_rad))
    
    # Ângulo do cotovelo esquerdo
    elbow_adjustment = math.radians(180 - cotovelo_esquerdo_angle)
    left_elbow_rad = left_shoulder_rad - elbow_adjustment
    
    left_wrist_x = int(left_elbow_x + forearm_length * math.cos(left_elbow_rad))
    left_wrist_y = int(left_elbow_y + forearm_length * math.sin(left_elbow_rad))
    
    # Desenhar braço esquerdo
    cv2.line(img, (left_shoulder_x, left_shoulder_y), (left_elbow_x, left_elbow_y), STICK_COLOR, 4)
    cv2.circle(img, (left_elbow_x, left_elbow_y), 5, JOINT_COLOR, -1)  # Articulação do cotovelo
    
    # Desenhar antebraço esquerdo
    cv2.line(img, (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y), STICK_COLOR, 4)
    cv2.circle(img, (left_wrist_x, left_wrist_y), 5, JOINT_COLOR, -1)  # Articulação do pulso
    
    # Adicionar texto de instrução baseado no exercício
    instructions = ""
    if exercicio_atual == "Agachamento":
        if etapa_atual == 0:
            instructions = "Posição inicial"
        else:
            instructions = "Flexione os joelhos"
    elif exercicio_atual == "Flexao":
        if etapa_atual == 0:
            instructions = "Posição alta"
        else:
            instructions = "Flexione os cotovelos"
    elif exercicio_atual == "Prancha":
        instructions = "Mantenha a posição"
    elif exercicio_atual == "Lunge":
        if etapa_atual == 0:
            instructions = "Posição inicial"
        elif etapa_atual == 1:
            instructions = "Perna direita à frente"
        else:
            instructions = "Perna esquerda à frente"
    elif exercicio_atual == "Deadlift":
        if etapa_atual == 0:
            instructions = "Posição inicial"
        else:
            instructions = "Incline-se para frente"
    elif exercicio_atual == "Shoulder Press":
        if etapa_atual == 0:
            instructions = "Cotovelos flexionados"
        else:
            instructions = "Estenda os braços"
    
    cv2.putText(img, instructions, (width//2 - 100, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    
    # Desenhar linha de base para percepção de solo
    ground_y = int(height * 0.8)
    cv2.line(img, (0, ground_y), (width, ground_y), (100, 100, 100), 2)
    
    return img

def process_camera():
    """
    Função atualizada para processar a câmera com tela dividida,
    mostrando um modelo 3D de referência.
    """
    global etapa_atual, ultimo_tempo_etapa, pontuacao, model_3d, exercicio_atual
    
    # Inicializar variáveis com DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ultimo_tempo_etapa = time.time()
    
    # Obter dimensões da captura
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        return
    
    height, width, _ = frame.shape
    
    # Reinicializar modelo 3D para garantir que está funcionando corretamente
    try:
        print("[INFO] Inicializando modelo 3D...")
        global model_3d
        model_3d = Model3D()  # Criar uma nova instância
        model_3d.init_opencv_surface(width, height)
        print("[INFO] Modelo 3D inicializado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar modelo 3D: {e}")
        # Salvar mensagem de erro em arquivo para depuração
        with open("erro_modelo3d.log", "w") as f:
            f.write(f"Erro ao inicializar modelo 3D: {str(e)}")
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Espelhar a imagem para visualização mais intuitiva
            frame = cv2.flip(frame, 1)
            
            # Processar frame com mediapipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Extrair ângulos do usuário
                user_angles = extract_angles_from_landmarks(results.pose_landmarks, width, height)
            
                # Obter valores de referência para a etapa atual
                reference_values = get_reference_pose()
            
                # Calcular pontuação baseada na etapa atual
                pontuacao = calcular_pontuacao(user_angles, reference_values)
                
                # Desenhar feedback na imagem
                frame = draw_reference_pose(frame, reference_values, user_angles)
                
                # Mostrar pontuação e etapa atual
                cv2.putText(frame, f"Pontuação: {pontuacao:.2f}", (20, height - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Etapa: {etapa_atual + 1}", (20, height - 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar o frame
            cv2.imshow('Exercício', frame)
            
            # Sair se pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_video(video_path, exercise_name):
    """
    Analisa um vídeo de exercício e fornece feedback sobre a postura.
    
    Args:
        video_path (str): Caminho para o arquivo de vídeo
        exercise_name (str): Nome do exercício a ser analisado
    """
    # Configurar o exercício atual
    set_exercicio(exercise_name)
    
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Não foi possível abrir o vídeo")
    
    # Obter informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Criar janela para exibição
    cv2.namedWindow("Análise de Exercício", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Análise de Exercício", width, height)
    
    # Variáveis para análise
    frame_count = 0
    total_score = 0
    feedback_history_video = [] # Renomeado para evitar conflito

    # Sistema de feedback em tempo real (para vídeo)
    feedback_system_video = {
        'current_feedback': [],
        'feedback_history': [],
        'score': 0,
        'frame_count': 0,
        'last_feedback_time': time.time(),
        'feedback_cooldown': 2.0,  # Tempo mínimo entre feedbacks (em segundos)
        'min_confidence': 0.6,     # Confiança mínima para considerar uma detecção válida
        'feedback_threshold': 3     # Número mínimo de frames com o mesmo feedback para exibi-lo
    }

    def update_feedback_system_video(frame_feedbacks, landmarks):
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

            if confidence < feedback_system_video['min_confidence']:
                return clean_text_for_display("Posicao nao detectada claramente. Ajuste sua posicao.")
        
        # Atualizar histórico de feedback
        if frame_feedbacks:
            feedback_system_video['feedback_history'].append(frame_feedbacks[0])
            # Manter apenas os últimos 30 feedbacks
            feedback_system_video['feedback_history'] = feedback_system_video['feedback_history'][-30:]
        
        # Verificar se passou tempo suficiente desde o último feedback
        if current_time - feedback_system_video['last_feedback_time'] < feedback_system_video['feedback_cooldown']:
            return None
        
        # Contar ocorrências do feedback mais recente
        if feedback_system_video['feedback_history']:
            recent_feedback = feedback_system_video['feedback_history'][-1]
            count = feedback_system_video['feedback_history'].count(recent_feedback)
            
            # Se o feedback se repetiu várias vezes, exibi-lo
            if count >= feedback_system_video['feedback_threshold']:
                feedback_system_video['last_feedback_time'] = current_time
                return recent_feedback
        
        return None

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

    def draw_feedback_box_video(frame, feedback, position, color):
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
                
            frame_count += 1
            
            # Processar o frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Desenhar landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Calcular ângulos
                lm = results.pose_landmarks.landmark
                joints = {
                    "Quadril": ([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                                [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                                [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]),
                    "Tronco": ([lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y], 
                               [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                               [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                    "Ombro Direito": ([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y], 
                                      [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], 
                                      [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
                    "Ombro Esquerdo": ([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y], 
                                       [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], 
                                       [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
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
                
                # Obter ângulos de referência
                reference_angles = get_reference_pose()
                
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
                frame_feedbacks = []
                frame_score = 0
                for name, angle in calculated_angles.items():
                    reference_angle = reference_angles.get(name, 0)
                    feedback, _ = avaliar_angulo(angle, reference_angle, name, exercise_name, etapa_atual, landmarks=results.pose_landmarks)
                    if feedback:
                        frame_feedbacks.append(feedback)
                        # O frame_score aqui será a pontuação da articulação, não apenas 1
                        # Em vez de frame_score, vamos usar o score retornado por calcular_pontuacao

                # Calcular a pontuação do frame usando a função calcular_pontuacao
                frame_score_calculated = calcular_pontuacao(calculated_angles, reference_angles)
                
                # Atualizar pontuação total (acumulando a pontuação calculada por frame)
                total_score += frame_score_calculated
                
                # Adicionar feedback ao histórico para o resumo final
                if frame_feedbacks:
                    feedback_history_video.append(frame_feedbacks[0])

                # Obter feedback em tempo real do sistema de feedback
                current_feedback_display = update_feedback_system_video(frame_feedbacks, results.pose_landmarks)

                # Desenhar feedback em tempo real
                if current_feedback_display:
                    draw_feedback_box_video(frame, 
                                            clean_text_for_display(current_feedback_display), 
                                            (width - 400, 100), 
                                            TURQUOISE)
                
                # Mostrar pontuação atual (usando a função de desenho da caixa)
                if frame_count > 0:
                    avg_score = total_score / frame_count
                    score_text = f"Pontuacao: {avg_score:.2f}"
                    draw_feedback_box_video(frame, 
                                            clean_text_for_display(score_text), 
                                            (width - 200, 50), 
                                            TURQUOISE)

                feedback_system_video['frame_count'] = frame_count # Atualizar contador de frames do sistema de feedback
                feedback_system_video['score'] = total_score # Atualizar pontuação do sistema de feedback

            # Mostrar o frame
            cv2.imshow("Análise de Exercício", frame)
            
            # Controlar a velocidade de reprodução
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar resumo final
    if feedback_history_video:
        # Criar janela de resumo
        summary_window = tk.Tk()
        summary_window.title(clean_text_for_display("Resumo da Analise"))
        summary_window.geometry("400x500")
        summary_window.configure(bg="#1A6FA3")
        
        # Título
        title_label = tk.Label(summary_window, 
                              text=clean_text_for_display("Resumo da Analise"),
                              font=("Helvetica", 20, "bold"),
                              fg="white",
                              bg="#1A6FA3")
        title_label.pack(pady=20)
        
        # Exercício
        exercise_label = tk.Label(summary_window,
                                text=clean_text_for_display(f"Exercicio: {exercise_name}"),
                                font=("Helvetica", 14),
                                fg="white",
                                bg="#1A6FA3")
        exercise_label.pack(pady=10)
        
        # Pontuação final
        score_label = tk.Label(summary_window,
                             text=clean_text_for_display(f"Pontuacao Final: {avg_score:.2f}"),
                             font=("Helvetica", 14),
                             fg="white",
                             bg="#1A6FA3")
        score_label.pack(pady=10)
        
        # Feedback mais comum
        if feedback_history_video:
            most_common_feedback = max(set(feedback_history_video), key=feedback_history_video.count)
            feedback_label = tk.Label(summary_window,
                                    text=clean_text_for_display("Feedback mais comum:"),
                                    font=("Helvetica", 14),
                                    fg="white",
                                    bg="#1A6FA3")
            feedback_label.pack(pady=10)
            
            feedback_text = tk.Text(summary_window,
                                  height=10,
                                  width=40,
                                  font=("Helvetica", 12),
                                  wrap=tk.WORD)
            feedback_text.pack(pady=10)
            feedback_text.insert(tk.END, clean_text_for_display(most_common_feedback))
            feedback_text.config(state=tk.DISABLED)
        
        # Botão para fechar
        close_button = tk.Button(summary_window,
                               text=clean_text_for_display("Fechar"),
                               command=summary_window.destroy,
                               font=("Helvetica", 12),
                               bg="#FFD600",
                               fg="#1A6FA3",
                               relief=tk.FLAT,
                               padx=20,
                               pady=10)
        close_button.pack(pady=20)
        
        summary_window.mainloop()

