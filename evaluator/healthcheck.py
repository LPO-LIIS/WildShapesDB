import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def validate_image(image_path):
    """Valida se a imagem pode ser carregada e processada corretamente"""

    # Transformações padrão
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])

    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        # Transformação para tensor
        img_tensor = transform(img)
        if img_tensor is None or img_tensor.shape[0] != 3:
            raise ValueError("Tensor inválido")

        return None  # Nenhum problema encontrado

    except (UnidentifiedImageError, OSError, ValueError) as e:
        return str(image_path)  # Retorna o caminho da imagem problemática


def compute_image_hash(image_path):
    """Computa um hash baseado no conteúdo exato da imagem."""
    try:
        img = Image.open(image_path).convert("RGB")  # Garantir 3 canais
        return hash(img.tobytes())  # Gera hash do conteúdo bruto
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None
    
def check_images(dataset_path, num_workers=8):
    """Verifica imagens corrompidas em todas as classes do dataset"""
    dataset_path = Path(dataset_path)
    problematic_images = []

    print(f"🔎 Verificando a integridade do dataset em: {dataset_path}")

    # Percorre todas as subpastas (classes)
    for class_folder in dataset_path.iterdir():
        if not class_folder.is_dir():  # Ignorar arquivos fora das pastas de classe
            continue

        print(f"\n📂 Verificando pasta da classe: {class_folder.name}")
        image_paths = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))

        # Processamento paralelo para verificar imagens mais rápido
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(validate_image, image_paths), total=len(image_paths), desc=f"{class_folder.name}", leave=False))

        # Filtra imagens corrompidas
        class_problematic = [img for img in results if img is not None]

        # Remover automaticamente as imagens ruins em lote
        for img in class_problematic:
            print(f"🗑️  Removendo {img}")
            os.remove(img)

        print("🔍 Verificando imagens dulicadas...")

        hashes = {}

        for image_path in tqdm(image_paths, desc=f"{class_folder.name}", leave=False):
            img_hash = compute_image_hash(image_path)
            if img_hash is None:
                continue

            # Verifica duplicatas exatas
            if img_hash in hashes:
                class_problematic.append(image_path)
                print(f"🗑️  Removendo duplicata exata: {image_path} (igual a {hashes[img_hash]})")
                os.remove(image_path)
            else:
                hashes[img_hash] = image_path

        problematic_images.extend(class_problematic)

        print(f"✅ Classe '{class_folder.name}' verificada. {len(class_problematic)} imagens removidas.")

    print(f"\n⚠️ Verificação concluída! Total de {len(problematic_images)} imagens problemáticas removidas.")