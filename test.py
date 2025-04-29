import os
import cv2
import torch
import torchvision.transforms as transforms

# Importar o modelo definido anteriormente
from classifier.model import Shape2DClassifier

# Configuração do dispositivo
device = torch.device("cpu")

# Carregar modelo treinado
model = Shape2DClassifier(
    num_classes=9
)  # Ajuste o número de classes conforme necessário
new_state_dict = torch.load("2dWildShapes.pth", map_location=device)
model.load_state_dict(new_state_dict) 
model.to(device)
model.eval()

# Transformação para preprocessamento das imagens
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)

# Dicionário de mapeamento de classes
class_names = [
    "Circle",
    "Ellipse",
    "Hexagon",
    "Parallelogram",
    "Pentagon",
    "Rectangle",
    "Square",
    "Trapezoid",
    "Triangle",
]

confidence_threshold = 0.7  # Limiar mínimo de confiança

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter imagem para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Aplicar transformações
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    # Fazer a inferência
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(
            output, dim=1
        )  # Converter logits para probabilidades
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    # Exibir resultado na imagem apenas se a confiança for maior que o limiar
    if confidence_score >= confidence_threshold:
        cv2.putText(
            frame,
            f"Prediction: {predicted_class} ({confidence_score:.2%})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Low Confidence",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Shape Classification", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
