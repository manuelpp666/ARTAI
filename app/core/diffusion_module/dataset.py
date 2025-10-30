import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset # ¡La biblioteca clave de Hugging Face!
from PIL import Image

class ArtImageDataset(Dataset):
    """
    Dataset para cargar imágenes de WikiArt y sus descripciones (prompts)
    usando la biblioteca 'datasets' de Hugging Face.
    """
    def __init__(self, tokenizer, size=512):
        print("Cargando dataset 'Artificio/WikiArt' desde Hugging Face...")
        # Descarga y carga el dataset. 
        # 'split="train"' usa el conjunto de entrenamiento.
        self.dataset = load_dataset("Artificio/WikiArt", split="train")
        
        self.tokenizer = tokenizer # El tokenizer de CLIP (se lo pasaremos en el train.py)
        self.size = size

        # ✅ ESTA ES LA NORMALIZACIÓN QUE PEDISTE:
        # Los modelos de difusión (como Stable Diffusion) esperan imágenes
        # normalizadas en el rango [-1, 1].
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(), # Convierte la imagen a [0, 1]
            transforms.Normalize([0.5], [0.5]), # Mapea de [0, 1] a [-1, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Obtener el item (imagen + texto)
        item = self.dataset[idx]
        
        # --- 1. Procesar la Imagen ---
        try:
            image = item['image'].convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            # Si una imagen está corrupta, cargamos la siguiente
            print(f"Error cargando imagen en índice {idx}, cargando siguiente: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # --- 2. Procesar el Texto (Prompt) ---
        # Usamos el campo 'description' que ya viene en este dataset
        prompt_text = item['description']
        
        # Tokenizar el prompt
        input_ids = self.tokenizer(
            prompt_text,
            padding="max_length", # Rellena/trunca a la longitud máxima
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids.squeeze() # Quitar dimensiones innecesarias
        }