import pygame
from pygame.locals import *
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2

class Model3D:
    def __init__(self):
        # Inicializar superfície de pygame para OpenGL
        self.width = 0
        self.height = 0
        self.surface = None
        self.initialized = False
        
        # Cores para diferentes partes do corpo
        self.body_color = (0.0, 0.8, 0.3, 1.0)  # Verde para o corpo
        self.joint_color = (1.0, 1.0, 0.0, 1.0)  # Amarelo para as articulações
        self.text_color = (1.0, 1.0, 1.0, 1.0)   # Branco para texto
        
        # Ângulos das articulações
        self.angles = {}
        
        # Visão lateralizada
        self.side_view = True
        
    def init_display(self, width, height):
        """Inicializa a exibição do pygame com OpenGL"""
        self.width = width
        self.height = height
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Configuração da perspectiva
        gluPerspective(45, (width / height), 0.1, 50.0)
        
        # Posição da câmera - vista lateral
        if self.side_view:
            gluLookAt(5, 2, 0,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        else:
            gluLookAt(0, 2, 5,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        
        self.initialized = True
        
    def init_opencv_surface(self, width, height):
        """Inicializa uma superfície para renderização do OpenGL para OpenCV"""
        self.width = width
        self.height = height
        
        pygame.init()
        self.surface = pygame.Surface((width, height))
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Configuração da perspectiva
        gluPerspective(45, (width / height), 0.1, 50.0)
        
        # Posição da câmera - vista lateral
        if self.side_view:
            gluLookAt(5, 2, 0,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        else:
            gluLookAt(0, 2, 5,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        
        self.initialized = True
        
    def set_angles(self, angles):
        """Define os ângulos das articulações"""
        self.angles = angles
        
    def draw_sphere(self, position, radius, color):
        """Desenha uma esfera em uma posição específica"""
        glColor4f(*color)
        sphere = gluNewQuadric()
        glPushMatrix()
        glTranslatef(*position)
        gluSphere(sphere, radius, 16, 16)
        glPopMatrix()
        gluDeleteQuadric(sphere)
        
    def draw_cylinder(self, start, end, radius, color):
        """Desenha um cilindro entre dois pontos"""
        glColor4f(*color)
        
        # Calcular direção e rotação
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        
        if length < 0.0001:
            return
        
        direction = direction / length
        
        # Criar um vetor perpendicular a direction
        up = np.array([0.0, 1.0, 0.0])
        
        # Se direction está próximo de up, use outro vetor
        if abs(np.dot(direction, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
            
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, direction)
        
        # Matriz de rotação
        matrix = np.eye(4)
        matrix[0, :3] = right
        matrix[1, :3] = up
        matrix[2, :3] = direction
        matrix[3, :3] = start
        
        # Desenhar cilindro
        glPushMatrix()
        glMultMatrixf(matrix.T)
        
        quadric = gluNewQuadric()
        gluCylinder(quadric, radius, radius, length, 10, 10)
        glPopMatrix()
        
        gluDeleteQuadric(quadric)
        
    def draw_human_model(self):
        """Desenha um modelo humano 3D"""
        # Tamanhos e proporções
        head_radius = 0.15
        joint_radius = 0.07
        body_radius = 0.08
        
        # Obter ângulos das articulações (ou usar valores padrão)
        torso_angle = math.radians(self.angles.get("Tronco", 90) - 90)
        
        right_shoulder_angle = math.radians(self.angles.get("Ombro Direito", 90) - 90)
        right_elbow_angle = math.radians(self.angles.get("Cotovelo Direito", 90) - 90)
        
        left_shoulder_angle = math.radians(self.angles.get("Ombro Esquerdo", 90) - 90)
        left_elbow_angle = math.radians(self.angles.get("Cotovelo Esquerdo", 90) - 90)
        
        right_hip_angle = math.radians(self.angles.get("Quadril Direito", 170) - 90)
        right_knee_angle = math.radians(self.angles.get("Joelho Direito", 170) - 90)
        
        left_hip_angle = math.radians(self.angles.get("Quadril Esquerdo", 170) - 90)
        left_knee_angle = math.radians(self.angles.get("Joelho Esquerdo", 170) - 90)
        
        # Para os ângulos do quadril, usamos o mesmo para ambos os lados
        hip_angle = math.radians(self.angles.get("Quadril", 170) - 90)
        
        # Ajustar ângulos baseados no exercício e visão lateral
        if self.side_view:
            # Rotação para visão lateral
            # A seção será ajustada com base no exercício atual
            pass
        
        # Posição da cabeça e pescoço
        head_pos = [0, 1.7, 0]
        neck_pos = [0, 1.5, 0]
        
        # Desenhar cabeça
        self.draw_sphere(head_pos, head_radius, self.body_color)
        
        # Posição da pelve e tronco
        pelvis_direction = [math.sin(torso_angle), -math.cos(torso_angle), 0]
        pelvis_pos = [
            neck_pos[0] + 0.6 * pelvis_direction[0],
            neck_pos[1] + 0.6 * pelvis_direction[1],
            neck_pos[2] + 0.6 * pelvis_direction[2]
        ]
        
        # Desenhar pescoço e tronco
        self.draw_cylinder(head_pos, neck_pos, body_radius, self.body_color)
        self.draw_sphere(neck_pos, joint_radius, self.joint_color)
        self.draw_cylinder(neck_pos, pelvis_pos, body_radius, self.body_color)
        self.draw_sphere(pelvis_pos, joint_radius, self.joint_color)
        
        # Ombro direito
        right_shoulder_pos = [
            neck_pos[0] + 0.25, 
            neck_pos[1], 
            neck_pos[2]
        ]
        self.draw_sphere(right_shoulder_pos, joint_radius, self.joint_color)
        
        # Cotovelo direito
        right_elbow_direction = [math.cos(right_shoulder_angle), math.sin(right_shoulder_angle), 0]
        right_elbow_pos = [
            right_shoulder_pos[0] + 0.3 * right_elbow_direction[0],
            right_shoulder_pos[1] + 0.3 * right_elbow_direction[1],
            right_shoulder_pos[2] + 0.3 * right_elbow_direction[2]
        ]
        self.draw_cylinder(right_shoulder_pos, right_elbow_pos, body_radius, self.body_color)
        self.draw_sphere(right_elbow_pos, joint_radius, self.joint_color)
        
        # Pulso direito
        right_hand_direction = [math.cos(right_elbow_angle), math.sin(right_elbow_angle), 0]
        right_hand_pos = [
            right_elbow_pos[0] + 0.3 * right_hand_direction[0],
            right_elbow_pos[1] + 0.3 * right_hand_direction[1],
            right_elbow_pos[2] + 0.3 * right_hand_direction[2]
        ]
        self.draw_cylinder(right_elbow_pos, right_hand_pos, body_radius, self.body_color)
        self.draw_sphere(right_hand_pos, joint_radius, self.joint_color)
        
        # Ombro esquerdo
        left_shoulder_pos = [
            neck_pos[0] - 0.25, 
            neck_pos[1], 
            neck_pos[2]
        ]
        self.draw_sphere(left_shoulder_pos, joint_radius, self.joint_color)
        
        # Cotovelo esquerdo
        left_elbow_direction = [math.cos(left_shoulder_angle), math.sin(left_shoulder_angle), 0]
        left_elbow_pos = [
            left_shoulder_pos[0] + 0.3 * left_elbow_direction[0],
            left_shoulder_pos[1] + 0.3 * left_elbow_direction[1],
            left_shoulder_pos[2] + 0.3 * left_elbow_direction[2]
        ]
        self.draw_cylinder(left_shoulder_pos, left_elbow_pos, body_radius, self.body_color)
        self.draw_sphere(left_elbow_pos, joint_radius, self.joint_color)
        
        # Pulso esquerdo
        left_hand_direction = [math.cos(left_elbow_angle), math.sin(left_elbow_angle), 0]
        left_hand_pos = [
            left_elbow_pos[0] + 0.3 * left_hand_direction[0],
            left_elbow_pos[1] + 0.3 * left_hand_direction[1],
            left_elbow_pos[2] + 0.3 * left_hand_direction[2]
        ]
        self.draw_cylinder(left_elbow_pos, left_hand_pos, body_radius, self.body_color)
        self.draw_sphere(left_hand_pos, joint_radius, self.joint_color)
        
        # Quadril direito
        right_hip_pos = [
            pelvis_pos[0] + 0.15, 
            pelvis_pos[1], 
            pelvis_pos[2]
        ]
        self.draw_sphere(right_hip_pos, joint_radius, self.joint_color)
        
        # Joelho direito
        right_knee_direction = [math.cos(right_hip_angle), math.sin(right_hip_angle), 0]
        right_knee_pos = [
            right_hip_pos[0] + 0.4 * right_knee_direction[0],
            right_hip_pos[1] + 0.4 * right_knee_direction[1],
            right_hip_pos[2] + 0.4 * right_knee_direction[2]
        ]
        self.draw_cylinder(right_hip_pos, right_knee_pos, body_radius, self.body_color)
        self.draw_sphere(right_knee_pos, joint_radius, self.joint_color)
        
        # Tornozelo direito
        right_ankle_direction = [math.cos(right_knee_angle), math.sin(right_knee_angle), 0]
        right_ankle_pos = [
            right_knee_pos[0] + 0.4 * right_ankle_direction[0],
            right_knee_pos[1] + 0.4 * right_ankle_direction[1],
            right_knee_pos[2] + 0.4 * right_ankle_direction[2]
        ]
        self.draw_cylinder(right_knee_pos, right_ankle_pos, body_radius, self.body_color)
        self.draw_sphere(right_ankle_pos, joint_radius, self.joint_color)
        
        # Quadril esquerdo
        left_hip_pos = [
            pelvis_pos[0] - 0.15, 
            pelvis_pos[1], 
            pelvis_pos[2]
        ]
        self.draw_sphere(left_hip_pos, joint_radius, self.joint_color)
        
        # Joelho esquerdo
        left_knee_direction = [math.cos(left_hip_angle), math.sin(left_hip_angle), 0]
        left_knee_pos = [
            left_hip_pos[0] + 0.4 * left_knee_direction[0],
            left_hip_pos[1] + 0.4 * left_knee_direction[1],
            left_hip_pos[2] + 0.4 * left_knee_direction[2]
        ]
        self.draw_cylinder(left_hip_pos, left_knee_pos, body_radius, self.body_color)
        self.draw_sphere(left_knee_pos, joint_radius, self.joint_color)
        
        # Tornozelo esquerdo
        left_ankle_direction = [math.cos(left_knee_angle), math.sin(left_knee_angle), 0]
        left_ankle_pos = [
            left_knee_pos[0] + 0.4 * left_ankle_direction[0],
            left_knee_pos[1] + 0.4 * left_ankle_direction[1],
            left_knee_pos[2] + 0.4 * left_ankle_direction[2]
        ]
        self.draw_cylinder(left_knee_pos, left_ankle_pos, body_radius, self.body_color)
        self.draw_sphere(left_ankle_pos, joint_radius, self.joint_color)
        
    def adjust_model_for_exercise(self, exercise, stage):
        """Ajusta o modelo 3D para diferentes exercícios"""
        if exercise == "Agachamento":
            if stage == 0:  # Posição inicial
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
            else:  # Posição agachada
                self.angles = {
                    "Tronco": 110,
                    "Quadril": 80,
                    "Joelho Direito": 80,
                    "Joelho Esquerdo": 80,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
        
        elif exercise == "Flexao":
            if stage == 0:  # Posição alta
                self.angles = {
                    "Tronco": 0,  # Paralelo ao chão
                    "Quadril": 170,
                    "Joelho Direito": 170,
                    "Joelho Esquerdo": 170,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 180,
                    "Ombro Esquerdo": 0
                }
            else:  # Posição baixa
                self.angles = {
                    "Tronco": 0,  # Paralelo ao chão
                    "Quadril": 170,
                    "Joelho Direito": 170,
                    "Joelho Esquerdo": 170,
                    "Cotovelo Direito": 90,
                    "Cotovelo Esquerdo": 90,
                    "Ombro Direito": 180,
                    "Ombro Esquerdo": 0
                }
        
        elif exercise == "Prancha":
            self.angles = {
                "Tronco": 0,  # Paralelo ao chão
                "Quadril": 170,
                "Joelho Direito": 170,
                "Joelho Esquerdo": 170,
                "Cotovelo Direito": 90,
                "Cotovelo Esquerdo": 90,
                "Ombro Direito": 180,
                "Ombro Esquerdo": 0
            }
        
        elif exercise == "Lunge":
            if stage == 0:  # Posição inicial
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
            elif stage == 1:  # Lunge direito
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 120,
                    "Joelho Direito": 90,
                    "Joelho Esquerdo": 165,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
            else:  # Lunge esquerdo
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 120,
                    "Joelho Direito": 165,
                    "Joelho Esquerdo": 90,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
        
        elif exercise == "Jumping Jacks":
            if stage == 0:  # Posição inicial
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 160,
                    "Cotovelo Esquerdo": 160,
                    "Ombro Direito": 90,
                    "Ombro Esquerdo": 90
                }
            else:  # Posição com braços e pernas abertos
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 120,
                    "Joelho Direito": 120,
                    "Joelho Esquerdo": 120,
                    "Cotovelo Direito": 175,
                    "Cotovelo Esquerdo": 175,
                    "Ombro Direito": 180,
                    "Ombro Esquerdo": 0
                }
        
        elif exercise == "Deadlift":
            if stage == 0:  # Posição inicial
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 175,
                    "Cotovelo Esquerdo": 175,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
            else:  # Posição baixa
                self.angles = {
                    "Tronco": 45,  # Inclinado para frente
                    "Quadril": 90,
                    "Joelho Direito": 140,
                    "Joelho Esquerdo": 140,
                    "Cotovelo Direito": 175,
                    "Cotovelo Esquerdo": 175,
                    "Ombro Direito": 30,
                    "Ombro Esquerdo": 150
                }
        
        elif exercise == "Shoulder Press":
            if stage == 0:  # Posição inicial
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 90,
                    "Cotovelo Esquerdo": 90,
                    "Ombro Direito": 60,
                    "Ombro Esquerdo": 120
                }
            else:  # Posição alta
                self.angles = {
                    "Tronco": 90,
                    "Quadril": 175,
                    "Joelho Direito": 175,
                    "Joelho Esquerdo": 175,
                    "Cotovelo Direito": 175,
                    "Cotovelo Esquerdo": 175,
                    "Ombro Direito": 170,
                    "Ombro Esquerdo": 10
                }
    
    def render_to_image(self, width, height, exercise="Agachamento", stage=0, countdown=0):
        """Renderiza o modelo 3D para uma imagem OpenCV"""
        if not self.initialized:
            self.init_opencv_surface(width, height)
        
        # Limpar a tela
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Redefinir matriz de visualização
        glLoadIdentity()
        
        # Posição da câmera - vista lateral
        if self.side_view:
            gluLookAt(5, 2, 0,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        else:
            gluLookAt(0, 2, 5,  # Posição da câmera
                     0, 1, 0,  # Ponto para onde olha
                     0, 1, 0)  # Vetor "para cima"
        
        # Ajustar modelo para o exercício atual
        self.adjust_model_for_exercise(exercise, stage)
        
        # Desenhar o modelo
        self.draw_human_model()
        
        # Desenhar o chão
        glColor4f(0.5, 0.5, 0.5, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-5, 0, -5)
        glVertex3f(-5, 0, 5)
        glVertex3f(5, 0, 5)
        glVertex3f(5, 0, -5)
        glEnd()
        
        # Obter a imagem da tela
        pygame.display.flip()
        
        # Ler os pixels do buffer
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Converter para numpy array
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        
        # Inverter a imagem verticalmente
        image = np.flipud(image)
        
        # Converter RGB para BGR (formato OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Adicionar informações ao rodapé
        cv2.putText(image, f"Exercício: {exercise}", (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Etapa: {stage + 1}", (20, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Adicionar contagem regressiva
        if countdown > 0:
            cv2.putText(image, f"Próxima posição em: {countdown:.1f}s", 
                      (width//2 - 150, height - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
        
    def cleanup(self):
        """Limpa os recursos do pygame"""
        pygame.quit() 