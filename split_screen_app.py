import cv2
import numpy as np
import time
from model_3d import Model3D
from pose_analysis import get_reference_pose, pontuacao, etapa_atual, exercicio_atual
import mediapipe as mp

def main():
    """
    Aplicativo simplificado que mostra a câmera à esquerda e o modelo 3D à direita
    """
    # Inicializar variáveis
    print("Iniciando aplicação com tela dividida...")
    
    # Captura de vídeo
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível acessar a câmera.")
        return
    
    height, width, _ = frame.shape
    
    # Inicializar modelo 3D
    try:
        model = Model3D()
        model.init_opencv_surface(width, height)
        print("Modelo 3D inicializado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar modelo 3D: {e}")
        cap.release()
        return
    
    # Configurar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Loop principal
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espelhar a imagem
            frame = cv2.flip(frame, 1)
            
            # Processar frame com MediaPipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Obter valores de referência atuais
            references = get_reference_pose()
            
            # Renderizar modelo 3D
            try:
                model_frame = model.render_to_image(width, height, exercicio_atual, etapa_atual, 3.0)
            except Exception as e:
                print(f"[ERRO] Falha ao renderizar modelo 3D: {e}")
                model_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(model_frame, "Erro ao renderizar modelo 3D", 
                          (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            
            # Criar tela dividida
            split_screen = np.zeros((height, width*2, 3), dtype=np.uint8)
            split_screen[:, :width] = frame
            split_screen[:, width:] = model_frame
            
            # Adicionar texto informativo
            cv2.putText(split_screen, "Sua pose", (50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(split_screen, "Modelo 3D", (width + 50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mostrar a tela dividida
            cv2.imshow("Tela Dividida", split_screen)
            
            # Sair se pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Limpar recursos
    cap.release()
    model.cleanup()
    cv2.destroyAllWindows()
    print("Aplicação encerrada.")

if __name__ == "__main__":
    main() 