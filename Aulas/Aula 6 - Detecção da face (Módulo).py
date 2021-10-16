################################################################################################
#                                                                                              #
#  Todo esse código será feito na extensão .py para que seja possível usá-lo nos projetos.     #
#                                                                                              #
#  Todas as anotações e explicações sobre o que está sendo usado nesse documento podem ser     #
#  encontradas no documento "Aula 5 - Detecção da face (Introdução).ipynb".                    #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


import cv2
import mediapipe as mp
import time


class DetectorRosto:
    def __init__(self, deteccao_confianca=0.5, alcance_deteccao=0):
        self.deteccao_confianca = deteccao_confianca
        self.alcance_deteccao = alcance_deteccao

        # Rosto
        self.mpDeteccaoRosto = mp.solutions.face_detection
        self.rosto = self.mpDeteccaoRosto.FaceDetection()

        # Desenhar o quadrado em volta do rosto
        self.mp_desenho = mp.solutions.drawing_utils

    def encontrar_rosto(self, imagem, desenho=True):   
        # Converter a cor da imagem (o Mediapipe usa somente imagens em RGB e o OpenCV captura em BGR)
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Resultado do processamento da imagem
        self.resultados = self.rosto.process(imagem_rgb)
        
        caixas_delimitadoras = []

        # Colocar o quadrado ao redor do rosto
        if self.resultados.detections:
            for item, deteccao in enumerate(self.resultados.detections):
                classe_caixa_delimitadora = deteccao.location_data.relative_bounding_box

                # Transformar as posições em pixels da tela
                altura, largura, canal = imagem.shape
                caixa_delimitadora = int(classe_caixa_delimitadora.xmin*largura),\
                int(classe_caixa_delimitadora.ymin*altura), int(classe_caixa_delimitadora.width*largura),\
                int(classe_caixa_delimitadora.height*altura)
                caixas_delimitadoras.append([item, caixa_delimitadora, deteccao.score])
                
                if desenho:
                    # Colocar o quadrado com as quinas grossas ao redor do rosto
                    imagem = self.quinas_grossas(imagem, caixa_delimitadora)

                    # Mostrar a porcentagem (score)
                    cv2.putText(imagem, f'{int(deteccao.score[0]*100)}%',
                                (caixa_delimitadora[0], caixa_delimitadora[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return imagem, caixas_delimitadoras
    
    def quinas_grossas(self, imagem, caixa, tamanho=30, espessura=5, espessura_retangulo=1):
        x, y, largura, altura = caixa
        x1, y1 = x + largura, y + altura
        
        # Desenhar o quadrado ao redor do rosto
        cv2.rectangle(imagem, caixa, (0, 255, 0), espessura_retangulo)
        
        # Desenhar a quina no canto superior esquerdo
        cv2.line(imagem, (x, y), (x+tamanho, y), (0, 255, 0), espessura)
        cv2.line(imagem, (x,y), (x, y + tamanho), (0, 255, 0), espessura)
        
        # Desenhar a quina no canto superior direito
        cv2.line(imagem, (x1, y), (x1 - tamanho, y), (0, 255, 0), espessura)
        cv2.line(imagem, (x1, y), (x1, y + tamanho), (0, 255, 0), espessura)
        
        # Desenhar a quina no canto inferior esquerdo
        cv2.line(imagem, (x, y1), (x + tamanho, y1), (0, 255, 0), espessura)
        cv2.line(imagem, (x, y1), (x, y1 - tamanho), (0, 255, 0), espessura)
        
        # Desenhar a quina no canto inferior direito
        cv2.line(imagem, (x1, y1), (x1 - tamanho, y1), (0, 255, 0), espessura)
        cv2.line(imagem, (x1, y1), (x1, y1 - tamanho), (0, 255, 0), espessura)
        
        return imagem


def main(video=0):
    cap = cv2.VideoCapture(video)
    
    detector = DetectorRosto()
    
    tempo_anterior = 0
    tempo_atual = 0
    
    while True:
        sucesso, imagem = cap.read()
        imagem, caixas_delimitadoras = detector.encontrar_rosto(imagem)
        
        # Configurar o FPS de captura
        tempo_atual = time.time()
        fps = 1/(tempo_atual - tempo_anterior)
        tempo_anterior = tempo_atual

        # Colocar o valor de FPS na tela
        cv2.putText(imagem, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Mostrar a imagem na tela
        cv2.imshow('Imagem', imagem)

        # Terminar o loop
        if cv2.waitKey(15) & 0xFF == ord('s'):
            break
        
    # Fechar a tela de captura
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
