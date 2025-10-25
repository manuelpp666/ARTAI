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
│   │   │   ├── train.py            # Entrenamiento desde cero o fine-tuning
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
│   ├── español/                    # Lenguaje Español
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
                        │   → Flask / Streamlit      │
                        └────────────┬───────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                                         │
       ┌────────▼────────┐                      ┌─────────▼──────────┐
       │   MÓDULO NLP    │                      │   MÓDULO VISUAL    │
       │ (Transformer)   │                      │ (Difusión/CNN)         │
       │ Comprende texto │                      │ Genera imágenes     │
       │ y crea prompts  │                      │ artísticas desde    │
       │ artísticos       │                     │ texto               │
       └────────┬────────┘                      └────────┬───────────┘
                │                                         │
   ┌────────────▼────────────┐             ┌──────────────▼─────────────┐
   │  Embeddings semánticos  │             │   Generación visual        │
   │  + Decoder Transformer  │             │   (UNet + DDPM o SD)       │
   │  (produce texto final o │             │   condicionado al prompt   │
   │   prompt visual)        │             │   textual del modelo NLP   │
   └────────────┬────────────┘             └──────────────┬─────────────┘
                │                                         │
                └────────────────┬────────────────────────┘
                                 │
                 ┌───────────────▼─────────────────┐
                 │     MÓDULO DE FUSIÓN / CONTROL  │
                 │ - Decide: responder o generar    │
                 │ - Coordina interacción texto↔imagen │
                 │ - Combina salida final (arte + texto)│
                 └─────────────────────────────────┘
                                 │
                      ┌──────────▼───────────┐
                      │  SALIDA AL USUARIO   │
                      │  (texto / imagen /   │
                      │   ambos)             │
                      └──────────────────────┘
