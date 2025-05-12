import cv2
import time
from model_3d import Model3D

def main():
    """Test the 3D model rendering separately"""
    print("Iniciando teste do modelo 3D...")
    
    try:
        # Inicializar modelo 3D
        model = Model3D()
        model.init_opencv_surface(640, 480)
        print("Modelo 3D inicializado com sucesso!")
        
        # Testar renderização
        for exercise in ["Agachamento", "Flexao", "Prancha", "Lunge", "Deadlift", "Shoulder Press"]:
            print(f"Testando renderização para exercício: {exercise}")
            
            for stage in range(2):  # Testar até 2 estágios para cada exercício
                print(f"  Estágio {stage+1}")
                
                try:
                    # Renderizar imagem
                    img = model.render_to_image(640, 480, exercise, stage, 3.0)
                    
                    if img is not None:
                        # Mostrar a imagem
                        cv2.imshow(f"{exercise} - Estágio {stage+1}", img)
                        cv2.waitKey(1000)  # Esperar 1 segundo
                        cv2.destroyAllWindows()
                        print("  OK!")
                    else:
                        print("  FALHA: A imagem renderizada é None")
                except Exception as e:
                    print(f"  ERRO ao renderizar: {e}")
        
        # Limpar recursos
        model.cleanup()
        print("Teste concluído!")
        
    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    main() 