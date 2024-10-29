# Proyecto de Filósofo Virtual

## Introducción

Bienvenido al proyecto de Filósofo Virtual, una iniciativa para entrenar un modelo de lenguaje con textos filosóficos. Este proyecto tiene como objetivo crear un modelo capaz de generar respuestas y análisis profundos basados en obras filosóficas clásicas y contemporáneas.

## Objetivos

- **Entrenamiento del modelo**: Utilizar textos filosóficos para ajustar un modelo de lenguaje preexistente.
- **Generación de contenido**: Crear un asistente virtual que pueda responder preguntas y generar contenido basado en filosofía.
- **Accesibilidad**: Hacer que el conocimiento filosófico sea más accesible a través de la tecnología.

## Tecnologías Utilizadas

- **Transformers**: Biblioteca de `Hugging Face` para modelos de lenguaje.
- **PyTorch**: Framework de aprendizaje profundo utilizado para entrenar el modelo.
- **PyPDF2**: Herramienta para extraer texto de archivos PDF.
- **Logging**: Configuración de registros para monitorear el proceso de entrenamiento.

## Estructura del Proyecto

1. **Extracción de Texto**: Utilizamos `PyPDF2` para extraer texto de archivos PDF de obras filosóficas.
2. **Tokenización**: Preprocesamos el texto utilizando `BloomTokenizerFast` para preparar los datos para el entrenamiento.
3. **Entrenamiento del Modelo**: Configuramos y entrenamos el modelo `BloomForCausalLM` con los textos tokenizados.
4. **Evaluación y Ajuste**: Evaluamos el rendimiento del modelo y ajustamos los parámetros según sea necesario.

## Script de Entrenamiento

Ver delphayTrain