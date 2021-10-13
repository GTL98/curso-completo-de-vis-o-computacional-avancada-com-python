################################################################################################
#                                                                                              #
#  Todo esse código será feito na extensão .py para que seja possível usá-lo nos projetos.     #
#                                                                                              #
#  Todas as anotações e explicações sobre o que está sendo usado nesse documento podem ser     #
#  encontradas no documento "Aula 3 - Rastramento de pose (Introdução).ipynb".                 #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


import cv2
import mediapipe as mp
import time


class DetectorPose:
    def __init__(self, modo=False, complexidade=1, suavidade_landmarks=True, segmentacao=False,
                 suavidade_segmentacao=True,
                 deteccao_confianca=0.5, rastreamento_confianca=0.5):
        self.modo = modo
        self.complexidade = complexidade
        self.suavidade_landmarks = suavidade_landmarks
        self.segmentacao = segmentacao
        self.suavidade_segmentacao = suavidade_segmentacao
        self.deteccao_confianca = deteccao_confianca
        self.rastreamento_confianca = rastreamento_confianca
        
        # Pose
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.modo, self.complexidade, self.suavidade_landmarks,
                                     self.segmentacao,
                                     self.suavidade_segmentacao, self.deteccao_confianca,
                                     self.rastreamento_confianca)

        # Desenhar as landmarks
        self.mp_desenho = mp.solutions.drawing_utils
        
    def encontrar_pose(self, imagem, fps, desenho=True):
        # Converter a cor da imagem (o Mediapipe usa somente imagens em RGB e o OpenCV captura em BGR)
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Resultado do processamento da imagem
        self.resultados = self.pose.process(imagem_rgb)
        
        # Colocar as landmarks no corpo
        if self.resultados.pose_landmarks:
            if desenho:
                self.mp_desenho.draw_landmarks(imagem, self.resultados.pose_landmarks,
                                               self.mpPose.POSE_CONNECTIONS,
                                         self.mp_desenho.DrawingSpec(color=(0, 0, 255)),
                                         self.mp_desenho.DrawingSpec(color=(0, 255, 0)))

        return imagem
                
    def encontrar_posicao(self, imagem, desenho=True):
        lista_landmark = []
        if self.resultados.pose_landmarks:
            for item, landmark in enumerate (self.resultados.pose_landmarks.landmark):
                # Pegar o valor do pixel onde cada landmark está
                altura, largura, canal = imagem.shape
                cx, cy = int(landmark.x*largura), int(landmark.y*altura)
                lista_landmark.append([item, cx, cy])
                if desenho:
                    cv2.circle(imagem, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    
        return lista_landmark
            
        
def main(video=0):
    tempo_anterior = 0
    tempo_atual = 0
    detector = DetectorPose()
    
    # Informar a pasta de onde está o vídeo. Se quiser usar a câmera, basta passar 0 como argumento ao invés da pasta do vídeo
    cap = cv2.VideoCapture(video)
    
    while True:
        sucesso, imagem = cap.read()
        
        # Configurar o FPS da captura
        tempo_atual = time.time()
        fps = 1 / (tempo_atual - tempo_anterior)
        tempo_anterior = tempo_atual
        
       
        imagem = detector.encontrar_pose(imagem, fps)
        lista_landmark = detector.encontrar_posicao(imagem)
        
        # Colcar o valor de FPS na tela
        cv2.putText(imagem, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

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
