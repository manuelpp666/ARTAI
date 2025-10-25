arte_chatbot/
│
├── app/                            # Núcleo del backend Flask
│   ├── routes/                     # Rutas de la API
│   │   └── chat.py                 # Endpoint principal del chat
│   │
│   ├── core/                       # Módulos de inteligencia
│   │   ├── nlp_module/             # Comprensión y generación de texto artístico
│   │   │   ├── preprocess.py       # Limpieza, tokenización, embeddings
│   │   │   ├── transformer.py      # Modelo Transformer entrenado
│   │   │   └── generator.py        # Respuesta generativa (estilo ChatGPT)
│   │   │
│   │   ├── diffusion_module/       # Generador de imágenes artísticas
│   │   │   ├── model.py            # Arquitectura del modelo de difusión
│   │   │   ├── train.py            # Entrenamiento desde cero o fine-tuning
│   │   │   └── generate.py         # Generación de imágenes a partir de texto
│   │   │
│   │   └── fusion_module/          # Integra texto + imagen coherente
│   │       └── fusion_engine.py    # Coordina qué generar según la intención
│   │
│   ├── utils/                      # Herramientas generales
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── helpers.py
│   │
│   ├── static/                     # Archivos estáticos para la interfaz web
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── templates/                  # HTML del chat
│       └── chat.html
│
├── datasets/                       # Datos para entrenamiento
│   ├── wikiart/                    # Imágenes por artista o estilo
│   ├── literatura/                 # Textos, poemas, descripciones de arte
│   └── prompts/                    # Prompts textuales para entrenamiento cruzado
│
├── models/                         # Modelos entrenados (.pt, .pkl)
│   ├── transformer_model.pt
│   ├── diffusion_model.pt
│   └── tokenizer.pkl
│
├── notebooks/                      # Experimentos, pruebas y prototipos
│   ├── 01_transformer_experiments.ipynb
│   ├── 02_diffusion_experiments.ipynb
│   └── 03_fusion_logic.ipynb
│
├── requirements.txt
└── README.md
│── main.py                     # Punto de entrada Flask (rutas + lógica del chat)





                        ┌────────────────────────────┐
                        │        INTERFAZ WEB        │
                        │  Chat + subida de imagen   │
                        │  + generación de arte/texto│
                        │   → Streamlit / Flask      │
                        └────────────┬───────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                                         │
       ┌────────▼────────┐                      ┌─────────▼──────────┐
       │   MÓDULO NLP    │                      │   MÓDULO VISUAL    │
       │ (Transformers)  │                      │ (Difusión / CNN)   │
       │ Comprende y     │                      │ Genera o reconoce  │
       │ genera texto     │                     │ imágenes artísticas │
       └────────┬────────┘                      └────────┬───────────┘
                │                                         │
   ┌────────────▼────────────┐             ┌──────────────▼─────────────┐
   │  Embeddings semánticos  │             │   Extracción o generación  │
   │  del texto (tokenizer + │             │   visual (UNet, DDPM)     │
   │  Transformer Encoder)   │             │   Condicionado al texto   │
   └────────────┬────────────┘             └──────────────┬─────────────┘
                │                                         │
      ┌─────────▼─────────┐                  ┌────────────▼───────────┐
      │ Decoder Transformer│                  │ Modelo de Difusión     │
      │ (Genera respuestas │                  │ (Genera arte o estilo  │
      │  textuales)        │                  │   desde ruido o texto) │
      └─────────┬─────────┘                  └────────────┬───────────┘
                │                                         │
                └────────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼─────────────────┐
                 │     MÓDULO DE FUSIÓN / CONTROL  │
                 │ - Fusiona texto e imagen         │
                 │ - Decide si responder, describir │
                 │   o generar arte                 │
                 │ - Controla prompts cruzados      │
                 └─────────────────────────────────┘
                                 │
                      ┌──────────▼───────────┐
                      │  SALIDA AL USUARIO   │
                      │  (texto / imagen /   │
                      │   ambos)             │
                      └──────────────────────┘
