import os  # <-- Asegúrate de que 'import os' esté aquí

# --- Mueve las otras importaciones aquí ---
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

class ArtImageDataset(Dataset):
    """
    Dataset para cargar imágenes de WikiArt y sus descripciones (prompts)
    usando la biblioteca 'datasets' de Hugging Face.
    """
    def __init__(self, tokenizer, size=512):
        
        # --- ¡ESTA ES LA SOLUCIÓN DEFINITIVA! ---
        
        # 1. Define tu ruta de caché aquí. 
        #    (Usa la ruta de la carpeta que SÍ tiene espacio en tu disco D:)
        cache_path = "D:\\cursos\\6to_ciclo\\inteligencia_artificial\\hugginface_cache\\datasets"
        
        # 2. (Opcional pero recomendado) Nos aseguramos que la carpeta exista
        os.makedirs(cache_path, exist_ok=True)
        
        # 3. Añadimos un print para que VEAS que está usando la ruta correcta
        print(f"--- ATENCIÓN: Usando directorio de caché explícito en: {cache_path} ---")
        
        # --- FIN DE LA SOLUCIÓN ---

        print("Cargando dataset 'Artificio/WikiArt' desde Hugging Face...")
        
        # 4. Modificamos 'load_dataset' para pasarle el argumento 'cache_dir'
        self.dataset = load_dataset(
            "Artificio/WikiArt",
            split="train",
            cache_dir=cache_path  # <--- ¡Aquí está la magia!
        )
        
        print("✅ Dataset cargado en la caché correcta.")
        
        self.tokenizer = tokenizer # El tokenizer de CLIP (se lo pasaremos en el train.py)
        self.size = size

        # ✅ ESTA ES LA NORMALIZACIÓN QUE PEDISTE:
        # (El resto de tu archivo no cambia)
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]), 
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # (El resto de tu archivo no cambia...)
        item = self.dataset[idx]
        
        try:
            image = item['image'].convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            print(f"Error cargando imagen en índice {idx}, cargando siguiente: {e}")
            return self.__getitem__((idx + 1) % len(self))

        prompt_text = item['description']
        
        input_ids = self.tokenizer(
            prompt_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids.squeeze()
        }