from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import BloomForCausalLM, BloomTokenizerFast
import logging
import PyPDF2, os, torch
import transformers.modeling_utils as hf

def init_log(log_filename):
    """
    Configura el logging para salida por consola y archivo .log.

    Args:
        log_filename (str): Nombre del archivo de log.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un archivo PDF.

    Args:
        pdf_path (str): Ruta del archivo PDF.

    Returns:
        str: Texto extraído del PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def load_training(train_folder, output_file):
    """
    Carga y procesa archivos PDF desde una carpeta, y guarda el texto extraído en un archivo de salida.

    Args:
        train_folder (str): Carpeta que contiene los archivos PDF.
        output_file (str): Archivo donde se guardará el texto extraído.
    """
    if os.path.exists(output_file):
        logging.info(f">> El archivo {output_file} ya existe. No se realizará ninguna acción.")
        return

    all_text = ''
    for filename in os.listdir(train_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(train_folder, filename)
            logging.info(f">> Procesando {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            logging.info(f">> Extraído {len(text)} caracteres")
            all_text += text + '\n'

    logging.info(f">> Guardando texto en {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text)

def compute_loss(model, inputs, num_items_in_batch, return_outputs=False):
    """
    Calcula la pérdida para el modelo.

    Args:
        model: El modelo de lenguaje.
        inputs: Los datos de entrada.
        num_items_in_batch: Número de elementos en el batch.
        return_outputs (bool): Si se deben devolver las salidas del modelo.

    Returns:
        torch.Tensor: La pérdida calculada.
    """
    labels = inputs.to(torch.long)
    outputs = model
    logits = outputs.logits

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

# Configurar logging para salida por consola y archivo .log
log_filename = 'training.log'
init_log(log_filename)

# Cargar el modelo y el tokenizador
model_name = "bigscience/bloom-560m"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
train_folder = 'ingest'
output_file = 'tokens.txt'

# Cargar y procesar datos de la carpeta de entrenamiento
load_training(train_folder, output_file)

# Cargar el dataset desde el archivo temporal
dataset = load_dataset('text', data_files={'train': output_file})

def tokenize_function(examples):
    """
    Tokeniza los ejemplos de texto.

    Args:
        examples (dict): Diccionario con los textos a tokenizar.

    Returns:
        dict: Diccionario con los textos tokenizados y las etiquetas.
    """
    tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

logging.info(f">> Comienza Entrenamiento")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    compute_loss_func=compute_loss
)

# Entrenar el modelo
try:
    trainer.train()
except Exception as e:
    logging.error(f"Error durante el entrenamiento: {e}")

# Guardar el modelo entrenado
trainer.save_model()

logging.info(f">> Entrenamiento terminado")
