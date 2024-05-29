# OpenCVGUI

Uma interface simples desenvolvida em Python com Tkinter demonstrando metodos comuns da biblioteca OpenCV

## Funcionalidades

A interface permite ativar e configurar uma série de funções de processamento de imagem, que podem ser combinadas sequencialmente de maneira dinâmica.

Todos os metodos de processamento estão contidos na classe [OpenCVUtils](/src/opencv_utils.py), que inclui:

- **Detecção de Face**: Utiliza o modelo [Face Mesh](/src/face_mesh_tracker.py) do mediapipe para detecção de 478 landmarks faciais

- **Detecção de Mão**: Utiliza o modelo [Hand Landmarker](/src/hand_tracker.py) do mediapipe para detecção de mãos

- **Filtro de cor**: Permite a configuração dos limites de cor HSV para a criação de uma máscara.

- **Detecção de Arestas**: Utiliza o metodo Canny com os limites inferior e superior.

- **Detecção de Contorno**: Utiliza o método findContours aplicado ao frame em escala de cinza com threshold.

- **Blur**: Aplica o método GaussianBlur com um kernel definido.

- **Rotação**: Permite a rotação da imagem em um ângulo específico.

- **Resize**: Permite alterar a altura e largura da imagem.

## Uso
Para utilizar o projeto, siga os passos abaixo:

Clone o repositório:

```bash
git clone https://github.com/Black-Bee-Drones/OpenCVGUI.git
```

Certifique-se de ter o Python instalado e instale as dependências necessárias:

```bash
python -m pip install -r requirements.txt
```

Execute o script main.py:

```bash
python src/main.py
```

## Observações

- Caso obtenha erro com os arquivos [.task](/res/), tente verificar as permissões dos arquivos:

    ```bash
    chmod +x res/hand_landamarker.task
    ```
    Ou exclua-os e execute o código novamente, o que fará o download dos arquivos novamente.

## Demonstração

A imagem a seguir demonstra a aplicação das funções de detecção de face, blur, detecção de arestas (Canny) e redimensionamento (resize).

![Screenshot from 2024-05-28 22-06-24](https://github.com/samuellimabraz/OpenCVGUI/assets/115582014/a42995a1-8497-4d87-a735-87f7cf7b5a03)


