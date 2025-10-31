import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator # Herramienta para manejar GPU/multi-GPU
import os
from tqdm.auto import tqdm # Para barras de progreso

# Importa tu clase Dataset del paso anterior
from dataset import ArtImageDataset 

# --- 1. Configuración ---
MODEL_ID = "CompVis/stable-diffusion-v1-4" # Modelo base para fine-tuning
DATASET_PATH = "Artificio/WikiArt" # Solo como referencia, ya está en tu dataset.py
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../../models/diffusion_art_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros de entrenamiento
IMG_SIZE = 512
BATCH_SIZE = 1 # ¡Importante! Empezar con 1 para no agotar la VRAM.
EPOCHS = 5
LEARNING_RATE = 1e-5

def main():
    # --- 2. Cargar Modelos y Tokenizer ---
    
    # El Tokenizer que tu Dataset necesita
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    
    # El codificador de texto (CLIP)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    
    # El Autoencoder (VAE) - Comprime imágenes a espacio latente
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    
    # El modelo U-Net (la "CNN" que vamos a entrenar)
    unet = UNet2DModel.from_pretrained(MODEL_ID, subfolder="unet")
    
    # El Scheduler - gestiona el proceso de ruido/des-ruido
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # --- 3. Preparar Dataset y DataLoader ---
    print("Cargando ArtImageDataset...")
    train_dataset = ArtImageDataset(tokenizer=tokenizer, size=IMG_SIZE)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    print(f"✅ Dataset cargado con {len(train_dataset)} ejemplos.")

    # --- 4. Preparar para Entrenamiento (Optimizador y Accelerator) ---
    
    # Solo entrenaremos la U-Net, congelamos lo demás
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train() # Asegurarse que la U-Net está en modo entrenamiento

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE
    )

    # Accelerator se encarga de mover todo al dispositivo (GPU)
    accelerator = Accelerator(
        mixed_precision='fp16' # Usar "mixed precision" para ahorrar memoria
    )
    unet, text_encoder, vae, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader
    )
    
    # Mover modelos a GPU (Accelerator ya lo hace, pero para estar seguros)
    device = accelerator.device
    vae.to(device)
    text_encoder.to(device)
    
    print(f"🚀 Iniciando entrenamiento en dispositivo: {device}")

    # --- 5. El Bucle de Entrenamiento ---
    
    global_step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(train_dataloader):
            # batch['pixel_values'] -> Imágenes ya normalizadas [-1, 1]
            # batch['input_ids'] -> Prompts tokenizados
            
            with torch.no_grad():
                # 1. Convertir imágenes a "latentes" (espacio comprimido)
                #    usando el VAE
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Codificar el prompt de texto
                text_embeddings = text_encoder(batch["input_ids"])[0]

            # 3. Generar ruido aleatorio
            noise = torch.randn_like(latents)
            
            # 4. Crear un "paso de tiempo" aleatorio (timestep)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
            
            # 5. Añadir ruido a las latentes (proceso de "difusión")
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- ¡El paso clave! ---
            # 6. Predecir el ruido usando la U-Net
            #    La U-Net ve la imagen ruidosa (noisy_latents) y
            #    usa el texto (text_embeddings) como "condición"
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # 7. Calcular la pérdida (Loss)
            #    Qué tan lejos estuvo nuestra predicción (noise_pred)
            #    del ruido real que añadimos (noise)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # 8. Backpropagation
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()

            # Actualizar barra de progreso
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

    # --- 6. Guardar el Modelo Entrenado ---
    print("✅ Entrenamiento finalizado.")
    
    # Guardar solo la U-Net, que es lo que hemos entrenado.
    # Usamos unwrap_model para obtener el modelo de PyTorch puro.
    final_unet = accelerator.unwrap_model(unet)
    final_unet.save_pretrained(OUTPUT_DIR)
    
    # (Opcional) También puedes guardar el pipeline completo
    # from diffusers import StableDiffusionPipeline
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     MODEL_ID,
    #     unet=final_unet,
    #     text_encoder=text_encoder,
    #     vae=vae,
    #     tokenizer=tokenizer
    # )
    # pipeline.save_pretrained(os.path.join(OUTPUT_DIR, "pipeline_completo"))

    print(f"💾 Modelo U-Net afinado guardado en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()