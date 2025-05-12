import cv2
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import time
import sys

class SimpleModel3D:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Modelo 3D")
        
        glEnable(GL_DEPTH_TEST)
        gluPerspective(45, (width / height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
        
        self.rotate_x = 0
        self.rotate_y = 0
        
    def draw_cube(self):
        vertices = (
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, 1)
        )
        
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 7),
            (3, 6)
        )
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        
    def draw_human(self):
        # Cabeça
        glPushMatrix()
        glTranslatef(0, 1.5, 0)
        glColor3f(0, 1, 0)  # Verde
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.25, 32, 32)
        glPopMatrix()
        
        # Torso (usando linhas em vez de cubo sólido)
        glPushMatrix()
        glTranslatef(0, 0.75, 0)
        glColor3f(0, 1, 0)  # Verde
        
        # Desenhar um retângulo para o torso
        glBegin(GL_QUADS)
        glVertex3f(-0.25, 0.5, 0.125)  # Canto superior esquerdo
        glVertex3f(0.25, 0.5, 0.125)   # Canto superior direito
        glVertex3f(0.25, -0.5, 0.125)  # Canto inferior direito
        glVertex3f(-0.25, -0.5, 0.125) # Canto inferior esquerdo
        glEnd()
        
        glPopMatrix()
        
        # Braços (usando linhas)
        glPushMatrix()
        glTranslatef(0, 1, 0)
        glColor3f(0, 0.8, 0)  # Verde mais escuro
        
        # Braço direito
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0.5, -0.3, 0)
        glEnd()
        
        # Braço esquerdo
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(-0.5, -0.3, 0)
        glEnd()
        
        glPopMatrix()
        
        # Pernas (usando linhas)
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glColor3f(0, 0.6, 0)  # Verde ainda mais escuro
        
        # Perna direita
        glBegin(GL_LINES)
        glVertex3f(0.2, 0.25, 0)
        glVertex3f(0.2, -0.7, 0)
        glEnd()
        
        # Perna esquerda
        glBegin(GL_LINES)
        glVertex3f(-0.2, 0.25, 0)
        glVertex3f(-0.2, -0.7, 0)
        glEnd()
        
        glPopMatrix()
        
    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
            # Rotação do modelo
            self.rotate_x += 1
            self.rotate_y += 1
            
            # Limpar a tela
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Aplicar rotação
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -5)
            glRotatef(self.rotate_x, 1, 0, 0)
            glRotatef(self.rotate_y, 0, 1, 0)
            
            # Desenhar o modelo 3D
            self.draw_human()
            
            # Atualizar display
            pygame.display.flip()
            pygame.time.wait(10)
            
        pygame.quit()
        sys.exit()

def main():
    print("Iniciando modelo 3D simples...")
    model = SimpleModel3D()
    model.render()

if __name__ == "__main__":
    main() 