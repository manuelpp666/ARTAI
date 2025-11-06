import bitsandbytes.optim as bnb_optim
import os
os.environ['HF_DATASETS_CACHE'] = 'D:\\cursos\\6to_ciclo\\inteligencia_artificial\\hugginface_cache\\datasets'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator # Herramienta para manejar GPU/multi-GPU
from tqdm.auto import tqdm # Para barras de progreso
from itertools import chain # Para el nuevo optimizer
import numpy as np
import argparse # <-- 1. IMPORTAMOS LA LIBRERÍA PARA EL SELECTOR

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
EPOCHS = 1
LEARNING_RATE = 1e-5

def main():

    # --- INICIO DEL CAMBIO 1: AÑADIR SELECTOR DE MODO ---
    parser = argparse.ArgumentParser(description="Entrenar modelo de difusión.")
    parser.add_argument(
        '--train_mode', 
        type=str, 
        default='all', 
        choices=['all', 'unet_only', 'text_encoder_only'],
        help='Qué parte del modelo entrenar: "all" (ambos), "unet_only" (solo difusor), o "text_encoder_only" (solo traductor)'
    )
    args = parser.parse_args()
    print(f"\n--- 🚀 MODO DE ENTRENAMIENTO SELECCIONADO: {args.train_mode} ---\n")
    # --- FIN DEL CAMBIO 1 ---


    # --- 2. Cargar Modelos y Tokenizer ---
    
    # El Tokenizer que tu Dataset necesita
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    
    # El codificador de texto (CLIP)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    
    # El Autoencoder (VAE) - Comprime imágenes a espacio latente
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    
    # El modelo U-Net (la "CNN" que vamos a entrenar)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    
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
    
    # --- INICIO DEL CAMBIO 2: LÓGICA DE CONGELACIÓN Y OPTIMIZADOR ---
    
    # VAE siempre congelado
    vae.requires_grad_(False)

    if args.train_mode == 'unet_only':
        print("INFO: Congelando text_encoder. Entrenando SOLO U-Net.")
        text_encoder.requires_grad_(False)
        unet.train()
        parametros_a_entrenar = unet.parameters()
        
    elif args.train_mode == 'text_encoder_only':
        print("INFO: Congelando U-Net. Entrenando SOLO text_encoder.")
        unet.requires_grad_(False)
        text_encoder.train()
        parametros_a_entrenar = text_encoder.parameters()
        
    else: # 'all'
        print("INFO: Entrenando U-Net y text_encoder (modo 'all').")
        unet.train()
        text_encoder.train()
        parametros_a_entrenar = chain(unet.parameters(), text_encoder.parameters())
    
    # --- FIN DEL CAMBIO 2 ---

    optimizer = bnb_optim.AdamW8bit(
        parametros_a_entrenar,
        lr=LEARNING_RATE
    )

    # Accelerator se encarga de mover todo al dispositivo (GPU)
    # Simula un batch size de 8 (1 imagen x 8 pasos)
    GRAD_ACCUM_STEPS = 8 

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=GRAD_ACCUM_STEPS
    )
    unet, text_encoder, vae, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader
    )
    
    # Mover modelos a GPU (Accelerator ya lo hace, pero para estar seguros)
    device = accelerator.device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device) # <--- Asegurarnos que U-Net también esté en el dispositivo
    
    print(f"🚀 Iniciando entrenamiento en dispositivo: {device}")

    # --- 5. El Bucle de Entrenamiento ---
    
    global_step = 0
    for epoch in range(EPOCHS):
        
        # Variables para métricas
        epoch_loss_total = 0.0
        batches_procesados = 0
        text_grad_indicator = float('nan') # Inicializar a 'nan' para mayor claridad

        # ✅ MEJORA (Tiempo): El total es el número de batches, no de pasos.
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
            #    (Si el text_encoder está congelado, esto no usará gradientes)
            text_embeddings = text_encoder(batch["input_ids"])[0]

            # 3. Generar ruido aleatorio
            noise = torch.randn_like(latents)
            
            # 4. Crear un "paso de tiempo" aleatorio (timestep)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=device).long()
            
            # 5. Añadir ruido a las latentes (proceso de "difusión")
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- ¡El paso clave! ---
            # 6. Predecir el ruido usando la U-Net
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # 7. Calcular la pérdida (Loss)
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # Acumular la pérdida de este batch
            epoch_loss_total += loss.item()
            batches_procesados += 1

            # 8. Backpropagation (solo calculará gradientes para los params no congelados)
            accelerator.backward(loss)
            
            # ✅ MEJORA (Tiempo): Actualizar la barra en CADA batch
            progress_bar.update(1)
            
            # El optimizador y la barra de progreso SÓLO deben actualizarse
            # cuando los gradientes se sincronizan (en el último paso de acumulación)
            if accelerator.sync_gradients:
                
                # --- ✅ CORRECCIÓN DE LÓGICA: Leemos el gradiente ANTES de step() y zero_grad() ---
                text_grad_indicator = float('nan') # Resetear a nan por si acaso
                if args.train_mode in ['all', 'text_encoder_only']:
                    # Desenvolvemos el modelo para acceder a los parámetros de PyTorch
                    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                    # Comprobamos el gradiente de la capa final
                    if unwrapped_text_encoder.text_model.final_layer_norm.weight.grad is not None:
                        # Calculamos el gradiente medio y lo guardamos
                        grad_val = unwrapped_text_encoder.text_model.final_layer_norm.weight.grad.mean().item()
                        if not np.isnan(grad_val): # Comprobar si no es NaN
                            text_grad_indicator = grad_val
                # --- Fin del Indicador ---

                optimizer.step()
                optimizer.zero_grad()
                
                # --- Métricas para la barra de progreso ---
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss_total / batches_procesados
                
                # Actualizar la barra con las nuevas métricas
                progress_bar.set_postfix(
                    avg_loss=f"{avg_loss:.5f}", 
                    lr=f"{current_lr:.8f}", # <-- LR en formato decimal
                    text_grad=f"{text_grad_indicator:.2e}" # <-- Indicador del traductor (mostrará 'nan' si no se entrena)
                )

                # Resetear contadores para el próximo ciclo de acumulación
                epoch_loss_total = 0.0
                batches_procesados = 0
            
            # ✅ MEJORA (Tiempo): Mostrar métricas intermedias mientras acumula
            else:
                if batches_procesados > 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_loss = epoch_loss_total / batches_procesados
                    progress_bar.set_postfix(
                        avg_loss=f"{avg_loss:.5f} (acum...)", 
                        lr=f"{current_lr:.8f}", # <-- LR en formato decimal
                        text_grad=f"{text_grad_indicator:.2e}" # <-- Indicador del traductor
                    )

    # --- 6. Guardar el Modelo Entrenado ---
    print("✅ Entrenamiento finalizado.")
    
    # --- INICIO DEL CAMBIO 3: LÓGICA DE GUARDADO ---
    
    # Guardar la U-Net solo si la entrenamos
    if args.train_mode in ['all', 'unet_only']:
        final_unet = accelerator.unwrap_model(unet)
        final_unet.save_pretrained(OUTPUT_DIR)
        print(f"💾 Modelo U-Net afinado guardado en: {OUTPUT_DIR}")
    
    # Guardar el Text Encoder solo si lo entrenamos
    if args.train_mode in ['all', 'text_encoder_only']:
        final_text_encoder = accelerator.unwrap_model(text_encoder)
        final_text_encoder.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))
        print(f"💾 Modelo Text Encoder afinado guardado en: {os.path.join(OUTPUT_DIR, 'text_encoder')}")
    
    # --- FIN DEL CAMBIO 3 ---

if __name__ == "__main__":
    main()