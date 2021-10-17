################################################################################################
#                                                                                              #
#  Todo esse código será feito na extensão .py para que seja possível usá-lo nos projetos.     #
#                                                                                              #
#  Todas as anotações e explicações sobre o que está sendo usado nesse documento podem ser     #
#  encontradas no documento "Aula 7 - Malha de rosto (Introdução).ipynb".                      #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


import cv2
import mediapipe as mp
import time


class DetectorMalhaRosto:
    def __init__(self, modo=False, max_rostos=2, deteccao_confianca=0.5, rastreamento_confianca=0.5):
        self.modo = modo
        self.max_rostos = max_rostos
        self.deteccao_confianca= deteccao_confianca
        self.rastreamento_confianca = rastreamento_confianca
        
        # Malha do rosto
        self.mpMalhaRosto = mp.solutions.face_mesh
        self.malha_rosto = self.mpMalhaRosto.FaceMesh(self.modo, self.max_rostos,
                                                      self.deteccao_confianca, self.rastreamento_confianca)

        # Desenhar a malha no rosto
        self.mp_desenho = mp.solutions.drawing_utils
        
    def encontrar_malha_rosto(self, imagem, desenho=True):
        # Converter a cor da imagem (o Mediapipe usa somente imagens em RGB e o OpenCV captura em BGR)
        self.imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Resultado do processamento da imagem
        self.resultados = self.malha_rosto.process(self.imagem_rgb)
        
        lista_rostos = []

        # Colocar a malha no rosto
        if self.resultados.multi_face_landmarks:
            for rosto_landmarks in self.resultados.multi_face_landmarks:
                if desenho:
                    self.mp_desenho.draw_landmarks(imagem, rosto_landmarks, self.mpMalhaRosto.FACEMESH_CONTOURS,
                                             self.mp_desenho.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                             self.mp_desenho.DrawingSpec(color=(0, 255, 0), thickness=1))

                lista_rosto = []
                for item, landmark in enumerate(rosto_landmarks.landmark):    
                    # Transformar os pontos em valores de pixel
                    altura, largura, canal = imagem.shape
                    x, y = int(landmark.x*largura), int(landmark.y*altura)
                    lista_rosto.append([x, y])
                    
                lista_rostos.append(lista_rosto)

        return imagem, lista_rostos


def main(video=0):
    cap = cv2.VideoCapture(video)
    
    tempo_anterior = 0
    tempo_atual = 0
    
    detector = DetectorMalhaRosto()
    
    while True:
        sucesso, imagem = cap.read()
        imagem, rostos = detector.encontrar_malha_rosto(imagem)
        
        # Configurar o FPS de captura
        tempo_atual = time.time()
        fps = 1/(tempo_atual - tempo_anterior)
        tempo_anterior = tempo_atual

        # Mostrar o FPS na tela
        cv2.putText(imagem, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Mostrar imagem na tela
        cv2.imshow('Imagem', imagem)

        # Terminar o loop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
            
    # Fechar a tela de captura
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
