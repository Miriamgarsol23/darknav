# DarkNav - Plan completo dias 6 al 10
# Miriam Garcia Sollo - June 2026

=============================================================
ESTADO ACTUAL (fin del dia 5)
=============================================================

Archivos existentes:
  src/synthetic.py       generador NFW completo + Dataset classes
  src/preprocessing.py   pipeline HDF5 -> .npy
  src/model.py           DarkNavUNet (scratch / imagenet / nfw)
  src/train.py           bucle de entrenamiento con CLI
  notebooks/00_scratch_day1.ipynb
  notebooks/01_data_exploration.ipynb
  notebooks/02_synthetic_generation.ipynb
  notebooks/03_model_and_training.ipynb
  data/synthetic/        1500 imagenes sinteticas generadas
  data/processed/        DEM patches reales filtrados

Problema pendiente: pos_weight=232 (mascaras muy escasas)
  -> ver seccion ANTES DE CONTINUAR abajo


=============================================================
ANTES DE CONTINUAR: arreglar las mascaras
=============================================================

El pos_weight=232 indica que las mascaras de target_masks en el
HDF5 son casi todas cero. Hay que inspeccionarlas:

  python3 -c "
  import h5py, numpy as np
  with h5py.File('data/train_images.hdf5','r') as f:
      keys = sorted(f['target_masks'].keys(), key=int)
      for k in keys[:5]:
          m = f['target_masks'][k][:]
          print(f'key={k} shape={m.shape} dtype={m.dtype} sum={m.sum()} max={m.max()}')
  "

Si max=0 en todas: las mascaras del HDF5 estan vacias.
Solucion: generar mascaras desde el CSV de cráteres en lugar de target_masks.

  python3 -c "
  import h5py, numpy as np
  with h5py.File('data/train_images.hdf5','r') as f:
      print('All keys:', list(f.keys()))
      if 'cll_xy' in f:
          k0 = sorted(f['cll_xy'].keys(), key=int)[0]
          print('cll_xy[0]:', f['cll_xy'][k0][:])
  "

cll_xy contiene las posiciones de los cráteres en coordenadas de pixel.
Si las mascaras estan vacias, usar cll_xy para regenerarlas:

  python src/preprocessing.py --data_dir data --out_dir data/processed --use_cll_xy

(Necesitaras anadir el flag --use_cll_xy a preprocessing.py si hace falta.
Dile a Claude que lo arregle mostrándole la salida del script de inspeccion.)


=============================================================
DIA 6 - Entrenamiento completo overnight
=============================================================

Comandos (ejecutar en orden, dejar correr):

  # Activar entorno
  source venv/bin/activate

  # Condicion A: scratch, 50 epocas
  python src/train.py --condition A --epochs 50 --out_dir runs
  # -> guarda: runs/condition_A/condition_A_best.pth
  # -> guarda: runs/condition_A/condition_A_history.json

  # Condicion B: imagenet pretrained, 50 epocas
  python src/train.py --condition B --epochs 50 --out_dir runs
  # -> guarda: runs/condition_B/condition_B_best.pth

  # Condicion C fase 1: preentrenamiento NFW, 30 epocas
  python src/train.py --condition C_pretrain --epochs 30 --out_dir runs
  # -> guarda: runs/condition_C/nfw_pretrain_best.pth

  # Condicion C fase 2: fine-tuning, 50 epocas
  python src/train.py --condition C_finetune --epochs 50 --out_dir runs \
      --nfw_checkpoint runs/condition_C/nfw_pretrain_best.pth
  # -> guarda: runs/condition_C/condition_C_best.pth

Entregable del dia 6:
  - 4 checkpoints .pth en runs/
  - 4 archivos _history.json con las curvas de entrenamiento
  - Commit: "day 6: full training complete, 4 conditions trained"

Tiempo estimado en CPU: 2-4 horas en total para los 130 epochs.
Puedes dejarlo corriendo en paralelo mientras haces otra cosa.

Si quieres ver el progreso en tiempo real:
  python src/train.py --condition A --epochs 50 --out_dir runs 2>&1 | tee logs/train_A.log


=============================================================
DIA 7 - Evaluacion cuantitativa
=============================================================

Prerequisito: los 3 checkpoints condition_X_best.pth existen.

Paso 1: instalar onnxruntime
  pip install onnxruntime

Paso 2: evaluar todo desde CLI
  python src/evaluate.py --proc_dir data/processed --runs_dir runs

Salida esperada:
  Condition      IoU   Precision  Recall      F1  CPU ms
  condition_A  0.XXXX    0.XXXX  0.XXXX  0.XXXX   XX.X
  condition_B  0.XXXX    0.XXXX  0.XXXX  0.XXXX   XX.X
  condition_C  0.XXXX    0.XXXX  0.XXXX  0.XXXX   XX.X

Guarda: runs/evaluation_results.json

Paso 3: abrir el notebook de evaluacion
  jupyter lab notebooks/04_evaluation_gradcam.ipynb

El notebook genera:
  docs/fig21_gradcam.png          mapas Grad-CAM comparados
  docs/fig22_inference_time.png   benchmark CPU
  docs/fig23_final_results.png    tabla de metricas

Commit:
  git add runs/evaluation_results.json runs/**/*.onnx
  git add docs/fig21* docs/fig22* docs/fig23*
  git add notebooks/04_evaluation_gradcam.ipynb
  git commit -m "day 7: full evaluation, grad-cam, onnx export"
  git push


=============================================================
DIA 8 - Analisis cualitativo y paper tecnico
=============================================================

Abrir el notebook 04_evaluation_gradcam.ipynb si no lo hiciste ayer.

Las figuras clave del paper son:
  fig9_real_vs_synthetic.png     analogia morfologica NFW-cráter
  fig13_generator_ablation.png   contribucion de cada componente del generador
  fig21_gradcam.png              donde mira cada modelo (el argumento central)
  fig23_final_results.png        tabla de resultados

Estructura del paper tecnico (docs/technical_report.md, ~6 paginas):

  1. Abstract (150 palabras)
     Problema: crater detection para TRN en deep space.
     Metodo: preentrenamiento con morfologia NFW proyectada.
     Resultado: Condition C mejora IoU en X% respecto a scratch.

  2. Introduction (0.5 pag)
     TRN en misiones sin GPS. Bottleneck: datos etiquetados escasos.
     Contribucion original: analogia NFW-cráter como fuente de datos sinteticos.

  3. Morphological analogy (0.5 pag)
     Ecuacion NFW. Proyeccion 2D. Figura fig9. Modelo de eyeccion.

  4. Dataset and preprocessing (0.5 pag)
     LOLA DEM. Zenodo 1133969. Filtros de calidad.
     1500 parches reales + 1500 sinteticos.

  5. Architecture (0.5 pag)
     U-Net con ResNet-18. Tabla de condiciones A/B/C.

  6. Results (1 pag)
     Tabla IoU/Prec/Recall/F1. Figura fig23. Figura fig21 Grad-CAM.
     Discusion: por que C supera a A y a B.

  7. On-orbit deployment (0.5 pag)
     Benchmark CPU. ONNX export. Contexto EnduroSat / FRAME.

  8. Conclusions (0.5 pag)
     NFW pretraining como regularizador geometrico.
     Lineas futuras: Marte, Ceres, Mercurio.

  9. References (usar bibliography.md)

Comandos para generar el paper:
  jupyter lab notebooks/04_evaluation_gradcam.ipynb
  # Completa las secciones del research_log con observaciones reales
  # Escribe docs/technical_report.md

Commit:
  git add docs/technical_report.md docs/research_log.md
  git commit -m "day 8: grad-cam analysis complete, technical report drafted"
  git push


=============================================================
DIA 9 - Limpieza, README final, ONNX benchmark
=============================================================

Paso 1: verificar que todos los notebooks corren de principio a fin
  jupyter nbconvert --to notebook --execute notebooks/00_scratch_day1.ipynb
  jupyter nbconvert --to notebook --execute notebooks/02_synthetic_generation.ipynb
  # Si alguno falla, arreglarlo antes de entregar

Paso 2: limpiar outputs de los notebooks (solo mantener outputs importantes)
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True \
      --to notebook notebooks/03_model_and_training.ipynb \
      --output notebooks/03_model_and_training.ipynb
  # Luego re-ejecutar para tener outputs limpios

Paso 3: README final
  Actualizar README.md con:
  - Resultados finales reales (no placeholders)
  - Instrucciones de instalacion verificadas
  - Enlace a cada notebook con descripcion de una linea
  - Tabla de resultados finales
  - Seccion "Connection to on-orbit deployment"

Paso 4: estructura final del repo
  darknav/
  ├── README.md               actualizado con resultados reales
  ├── requirements.txt
  ├── .gitignore
  ├── docs/
  │   ├── design_doc.md
  │   ├── bibliography.md     25 fuentes
  │   ├── research_log.md     entradas de los 10 dias
  │   ├── technical_report.md el paper de 6 paginas
  │   └── fig*.png            23 figuras
  ├── src/
  │   ├── synthetic.py
  │   ├── preprocessing.py
  │   ├── model.py
  │   ├── train.py
  │   └── evaluate.py
  ├── notebooks/
  │   ├── 00_scratch_day1.ipynb
  │   ├── 01_data_exploration.ipynb
  │   ├── 02_synthetic_generation.ipynb
  │   ├── 03_model_and_training.ipynb
  │   └── 04_evaluation_gradcam.ipynb
  ├── runs/
  │   ├── condition_A/
  │   ├── condition_B/
  │   └── condition_C/
  └── data/                   gitignored
      ├── synthetic/
      └── processed/

Paso 5: tag de version
  git add .
  git commit -m "day 9: clean notebooks, final README, repo structure complete"
  git tag v1.0.0
  git push && git push --tags


=============================================================
DIA 10 - Entrega final
=============================================================

Checklist antes de entregar:

Documentacion:
  [ ] README.md tiene resultados reales, no placeholders
  [ ] bibliography.md tiene 20+ referencias con DOI
  [ ] research_log.md tiene entradas para los 10 dias
  [ ] technical_report.md tiene 4-6 paginas con ecuaciones y figuras
  [ ] design_doc.md esta actualizado con las decisiones reales tomadas

Codigo:
  [ ] Los 5 notebooks corren sin errores desde cero
  [ ] src/ tiene los 5 modulos Python con docstrings
  [ ] requirements.txt esta completo y verificado
  [ ] .gitignore excluye data/ y checkpoints grandes

Evidencia de proceso (lo mas importante):
  [ ] Git log tiene commits con fechas reales dia a dia
  [ ] research_log.md muestra lectura real de papers con anotaciones propias
  [ ] docs/ tiene 20+ figuras generadas por los notebooks
  [ ] runs/ tiene los historiales JSON de entrenamiento (curvas reales)
  [ ] El notebook 00 tiene la figura NFW-cráter con cross-sections

Narrativa de proyecto:
  [ ] El README conecta el proyecto con TRN y con el contexto de hardware limitado
  [ ] El technical_report tiene una seccion explicando la analogia NFW-cráter
  [ ] El Grad-CAM figura muestra que el modelo C enfoca el rim circular

Comandos para verificacion final:
  # Verificar que el repo esta limpio
  git status
  git log --oneline | head -20

  # Verificar que los checkpoints existen
  find runs/ -name "*.pth" | sort

  # Verificar las figuras
  find docs/ -name "*.png" | sort | wc -l  # deberia ser 20+

  # Hacer un test rapido de inferencia
  python3 -c "
  import torch
  from src.model import DarkNavUNet
  m = DarkNavUNet(mode='scratch')
  x = torch.zeros(1,1,128,128)
  with torch.no_grad():
      y = m(x)
  print('Model output shape:', y.shape)
  print('Everything works.')
  "

Entrega:
  URL del repositorio GitHub publico.
  El evaluador (Krasimir Stoev) puede clonar y ejecutar:
    git clone https://github.com/TU_USUARIO/darknav
    cd darknav
    pip install -r requirements.txt
    jupyter lab


=============================================================
RESUMEN DE TODOS LOS COMANDOS EN ORDEN
=============================================================

# Dia 6
source venv/bin/activate
python src/train.py --condition A --epochs 50 --out_dir runs
python src/train.py --condition B --epochs 50 --out_dir runs
python src/train.py --condition C_pretrain --epochs 30 --out_dir runs
python src/train.py --condition C_finetune --epochs 50 --out_dir runs \
    --nfw_checkpoint runs/condition_C/nfw_pretrain_best.pth
git add runs/ && git commit -m "day 6: full training" && git push

# Dia 7
pip install onnxruntime
python src/evaluate.py --proc_dir data/processed --runs_dir runs
# abrir 04_evaluation_gradcam.ipynb y ejecutar todo
git add . && git commit -m "day 7: evaluation and gradcam" && git push

# Dia 8
# escribir docs/technical_report.md
# completar research_log.md con dias 6-8
git add docs/ && git commit -m "day 8: technical report" && git push

# Dia 9
pip install nbconvert
jupyter nbconvert --to notebook --execute notebooks/02_synthetic_generation.ipynb
# verificar todos los notebooks
git add . && git commit -m "day 9: final cleanup" && git push
git tag v1.0.0 && git push --tags

# Dia 10
git log --oneline
find docs/ -name "*.png" | wc -l
find runs/ -name "*.pth" | sort
