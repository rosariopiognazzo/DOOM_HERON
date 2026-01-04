# File WAD per VizDoom
# =====================
#
# I file .wad sono i file di scenario di VizDoom e NON possono essere creati
# manualmente - devono essere scaricati dalla libreria VizDoom.
#
# ISTRUZIONI PER OTTENERE I FILE WAD:
# -----------------------------------
#
# 1. Installa VizDoom:
#    pip install vizdoom
#
# 2. I file .wad sono inclusi nel pacchetto VizDoom.
#    Puoi trovarli nella directory di installazione:
#    
#    Python:
#    >>> import vizdoom
#    >>> import os
#    >>> print(os.path.dirname(vizdoom.__file__))
#    
#    I file .wad si trovano in: <vizdoom_path>/scenarios/
#
# 3. Copia i seguenti file in questa cartella (doom_files/):
#    - basic.wad
#    - deadly_corridor.wad
#    - defend_the_center.wad
#
# ALTERNATIVA: Modifica il config_path nel codice per puntare alla
# cartella scenarios di VizDoom:
#
#    import vizdoom
#    import os
#    config_path = os.path.join(os.path.dirname(vizdoom.__file__), 'scenarios')
#
# Esempio di codice per copiare automaticamente i file:
# -----------------------------------------------------
#
# import vizdoom
# import shutil
# import os
#
# vizdoom_path = os.path.dirname(vizdoom.__file__)
# scenarios_path = os.path.join(vizdoom_path, 'scenarios')
# target_path = 'vizdoom/doom_files'
#
# for scenario in ['basic', 'deadly_corridor', 'defend_the_center']:
#     src = os.path.join(scenarios_path, f'{scenario}.wad')
#     if os.path.exists(src):
#         shutil.copy(src, target_path)
#         print(f'Copied {scenario}.wad')
