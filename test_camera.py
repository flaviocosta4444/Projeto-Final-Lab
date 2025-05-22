import cv2

def test_camera():
    print("Testando acesso à câmera...")
    
    # Tentar diferentes índices de câmera
    for camera_index in range(3):
        print(f"\nTentando câmera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Não foi possível abrir a câmera {camera_index}")
            continue
            
        print(f"Câmera {camera_index} aberta com sucesso!")
        ret, frame = cap.read()
        
        if ret:
            print(f"Frame capturado com sucesso! Dimensões: {frame.shape}")
            cv2.imshow(f'Camera {camera_index}', frame)
            cv2.waitKey(2000)  # Mostrar por 2 segundos
        else:
            print("Não foi possível capturar frame")
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera() 