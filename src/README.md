# src layout

Questa cartella contiene il codice sorgente modulare del training DPO e i moduli importati.

## Struttura

- `DPO_train.py`: orchestratore del training/eval.
- `dpo_config.py`: configurazione centralizzata (`Config`, `CONFIG`).
- `dpo_data.py`: lettura FASTA, padding/encoding, dataset e pair dataset.
- `dpo_metrics.py`: loss DPO e metriche NLL/likelihood/correlazione.
- `dpo_plotting.py`: generazione dei plot per epoca.
- `dpo_logging.py`: stampa ordinata dello standard output.
- `transformer.py`: definizione modello `GPTTransformer`.
- `Transformer_Reint.py`: utilita' dataset legacy (`load_dataset`, etc.).

## Esecuzione

Da `Transformer/` puoi usare uno dei due comandi:

- `python3 DPO_train.py` (launcher compatibile in root)
- `python3 -m src.DPO_train`

Entrambi usano la stessa configurazione in `src/dpo_config.py`.
