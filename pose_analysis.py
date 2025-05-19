import cv2
import mediapipe as mp
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_3d import Model3D  # Import the 3D model class
import os

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

def draw_angle(image, point, angle, feedback, color=(0, 255, 0), y_offset=0):
    """
    Desenha o ângulo e o feedback na imagem.
    """
    h, w = image.shape[:2]
    x, y = int(point[0] * w), int(point[1] * h)
    
    # Desenhar o ângulo
    cv2.putText(
        image,
        f"{int(angle)}°",
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
            feedback,
            (10, 30 + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

def avaliar_angulo(angulo_atual, angulo_referencia, nome_articulacao, exercicio_atual, etapa_atual):
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
        }
    }
    
    # Obter a margem de tolerância para a articulação atual
    margem = margens.get(exercicio_atual, {}).get(nome_articulacao, 15)
    
    # Calcular a diferença absoluta
    diferenca = abs(angulo_atual - angulo_referencia)
    
    # Gerar feedback específico para cada exercício e etapa
    feedback = ""
    if exercicio_atual == "Flexao":
        if nome_articulacao in ["Cotovelo Direito", "Cotovelo Esquerdo"]:
            if etapa_atual == 0:  # Posição inicial
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Estenda mais o braço"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Mantenha os braços estendidos"
            else:  # Posição baixa
                if angulo_atual < angulo_referencia - margem:
                    feedback = f"{nome_articulacao}: Desça mais"
                elif angulo_atual > angulo_referencia + margem:
                    feedback = f"{nome_articulacao}: Mantenha a posição baixa"
        elif nome_articulacao == "Tronco":
            if angulo_atual > angulo_referencia + margem:
                feedback = "Mantenha o corpo reto"
            elif angulo_atual < angulo_referencia - margem:
                feedback = "Não arqueie as costas"
        elif nome_articulacao == "Quadril":
            if angulo_atual > angulo_referencia + margem:
                feedback = "Mantenha o quadril alinhado"
            elif angulo_atual < angulo_referencia - margem:
                feedback = "Não deixe o quadril cair"
    
    return feedback, (255, 255, 255)  # Retorna feedback e cor branca

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
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 1: Posição agachada
                "Joelho Direito": 80,
                "Joelho Esquerdo": 80,
                "Quadril": 80,
                "Tronco": 90,
                "Cotovelo Direito": 160,
                "Cotovelo Esquerdo": 160,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
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
        "Deadlift": [
            {  # Etapa 0: Posição inicial
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Quadril": 175,
                "Tronco": 90,
                "Cotovelo Direito": 175,
                "Cotovelo Esquerdo": 175,
                "Ombro Direito": 90,
                "Ombro Esquerdo": 90
            },
            {  # Etapa 1: Posição baixa
                "Joelho Direito": 140,
                "Joelho Esquerdo": 140,
                "Quadril": 90,
                "Tronco": 45,  # Inclinado para frente
                "Cotovelo Direito": 175,
                "Cotovelo Esquerdo": 175,
                "Ombro Direito": 130,
                "Ombro Esquerdo": 50
            }
        ],
        "Shoulder Press": [
            {  # Etapa 0: Posição inicial
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Quadril": 175,
                "Tronco": 90,
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Ombro Direito": 60,
                "Ombro Esquerdo": 120
            },
            {  # Etapa 1: Posição alta
                "Cotovelo Direito": 175,
                "Cotovelo Esquerdo": 175,
                "Quadril": 175,
                "Tronco": 90,
                "Joelho Direito": 175,
                "Joelho Esquerdo": 175,
                "Ombro Direito": 170,
                "Ombro Esquerdo": 10
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
        "Agachamento": 2,
        "Flexao": 3,
        "Prancha": 1,
        "Lunge": 3,
        "Deadlift": 2,
        "Shoulder Press": 2
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
    Calcula a pontuação com base na semelhança entre os ângulos do usuário 
    e os valores de referência.
    """
    if not reference_values or not user_joints:
        return 0
    
    total_diff = 0
    count = 0
    
    # Para cada ângulo de referência, calcular a diferença com o ângulo do usuário
    for joint_name, ref_angle in reference_values.items():
        if joint_name in user_joints:
            user_angle = user_joints[joint_name]
            diff = abs(user_angle - ref_angle)
            
            # Limitar a diferença para não penalizar demais
            diff = min(diff, 45)
            total_diff += diff
            count += 1
    
    if count == 0:
        return 0
    
    # Calcular pontuação de 0 a 100
    avg_diff = total_diff / count
    score = max(0, 100 - (avg_diff * 100 / 45))
    
    return round(score)

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
        user_angle = user_angles.get(joint_name, 0)
        diff = abs(user_angle - ref_angle)
        
        # Determinar cor com base na diferença
        cor = (0, 255, 0)  # Verde se estiver próximo
        if diff > 15:
            cor = (255, 255, 0)  # Amarelo se houver diferença moderada
        if diff > 30:
            cor = (255, 0, 0)  # Vermelho se houver grande diferença
        
        texto = f"{joint_name}: {ref_angle}° (você: {user_angle}°)"
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
            
            # Calcular o tempo restante até a próxima etapa
            tempo_atual = time.time()
            tempo_restante = max(0, TEMPO_POR_ETAPA - (tempo_atual - ultimo_tempo_etapa))
            
            # Atualizar etapa do exercício
            atualizar_etapa()
            
            # Obter valores de referência para a etapa atual
            reference_values = get_reference_pose()
            
            # Criar imagem dividida (frame original à esquerda, modelo 3D de referência à direita)
            display_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Processar frame com mediapipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Inicializar dicionário para armazenar ângulos calculados
            calculated_angles = {}
            # Lista para feedbacks principais
            main_feedbacks = []
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
                for name, (a, b, c) in joints.items():
                    a = [a[0] * width, a[1] * height]
                    b = [b[0] * width, b[1] * height]
                    c = [c[0] * width, c[1] * height]
                    angle = calculate_angle(a, b, c)
                    calculated_angles[name] = angle
                    feedback, color = avaliar_angulo(angle, angle, name, exercicio_atual, etapa_atual)
                    draw_angle(frame, b, angle, feedback, color=color, y_offset=y_offset)
                    y_offset += 30
                    # Guardar feedbacks principais para agachamento
                    if exercicio_atual == "Agachamento":
                        if name == "Quadril" or name == "Tronco" or "Ombro" in name:
                            main_feedbacks.append(feedback)
                # Calcular pontuação
                pontuacao = calcular_pontuacao(calculated_angles, reference_values)
                # Desenhar informações de referência
                draw_reference_pose(frame, reference_values, calculated_angles)
            # Exibir feedbacks principais no topo da tela
            if main_feedbacks:
                for i, fb in enumerate(main_feedbacks):
                    cv2.putText(frame, fb, (20, 40 + i*35), (cv2.FONT_HERSHEY_SIMPLEX), 1.1, (0, 255, 255), 3)
            
            # Renderizar modelo 3D ou usar boneco de palito como fallback
            try:
                if model_3d and model_3d.initialized:
                    model_3d_frame = model_3d.render_to_image(width, height, exercicio_atual, etapa_atual, tempo_restante)
                    print(f"[INFO] Modelo 3D renderizado para {exercicio_atual}, etapa {etapa_atual}")
                else:
                    raise Exception("Modelo 3D não inicializado")
                    
            except Exception as e:
                print(f"[ERRO] Falha ao renderizar modelo 3D: {e}")
                # Usar boneco de palito como fallback
                model_3d_frame = create_stick_figure(reference_values, width, height)
                # Adicionar contagem regressiva
                countdown_text = f"Próxima posição em: {tempo_restante:.1f}s"
                cv2.putText(model_3d_frame, countdown_text, (width//2 - 150, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Adicionar dicas de posicionamento para cada articulação
            if len(calculated_angles) > 0:
                dicas = []
                for joint_name, ref_angle in reference_values.items():
                    if joint_name in calculated_angles:
                        user_angle = calculated_angles[joint_name]
                        diff = user_angle - ref_angle
                        if abs(diff) > 20:
                            if diff > 0:
                                dicas.append(f"{joint_name}: Flexione mais")
                            else:
                                dicas.append(f"{joint_name}: Estenda mais")
                
                # Mostrar até 3 dicas mais importantes
                dicas = dicas[:3]
                if dicas:
                    cv2.putText(frame, "Dicas para melhorar:", (10, height - 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    for i, dica in enumerate(dicas):
                        cv2.putText(frame, dica, (10, height - 70 + i*25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Colocar o frame original à esquerda
            display_frame[:, :width] = frame
            
            # Verificar se model_3d_frame é None
            if model_3d_frame is None:
                model_3d_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(model_3d_frame, "Erro ao renderizar modelo 3D", 
                          (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            
            # Colocar o modelo (3D ou boneco de palito) à direita
            display_frame[:, width:] = model_3d_frame
            
            # Adicionar texto para orientação
            cv2.putText(display_frame, "Sua pose", (50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Modelo de referência (vista lateral)", (width + 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Adicionar pontuação grande no centro inferior da tela
            score_text = f"Pontuação: {pontuacao}%"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (width * 2 - text_size[0]) // 2
            cv2.putText(display_frame, score_text, (text_x, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Mostrar tela dividida
            cv2.imshow('Siga o Modelo', display_frame)
            
            # Sair quando pressionar 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Limpeza
    cap.release()
    try:
        if model_3d and model_3d.initialized:
            model_3d.cleanup()
    except Exception as e:
        print(f"[ERRO] Falha ao limpar modelo 3D: {e}")
    cv2.destroyAllWindows()

