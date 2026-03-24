# RNA-GPT-DPO

## Criteri di nomenclatura dei file di pairing

I file di pairing sono tutti in formato `.csv` e sono organizzati per split:

- `pair_train/`: pairing usati nel training
- `pair_validation/`: pairing usati nella validazione

Per ogni sequenza funzionale del validation, c'e una riga nel CSV con i pairing verso sequenze non funzionali che rispettano un criterio specifico.

### Struttura del nome file

Formato generale:

`top{K}_{spazio}[_refbin][_reciprocal].csv`

Esempi:

- `top20_nt.csv`
- `top100_bin_refbin.csv`
- `top500_nt_reciprocal.csv`
- `top5_bin_refbin_reciprocal.csv`

### Significato delle parti

- `top{K}`: usa i `K` vicini piu prossimi (k-nearest neighbors)
- `nt`: distanza nello spazio dei nucleotidi
- `bin`: distanza nello spazio delle posizioni in comune con Azoarcus
- `_refbin`: i vicini sono cercati solo dentro il reference bin (stessa classe/bin di distanza da Azoarcus)
- `_reciprocal`: oltre ai vicini della sequenza funzionale, si aggiungono anche i vicini "reciproci"

### Cosa significa reciprocal (in pratica)

Se una sequenza non funzionale `i` ha tra i suoi vicini una funzionale `j`, allora `i` viene aggiunta ai vicini di `j` (se non era gia presente).

In questo modo si ottiene un pairing piu simmetrico tra funzionali e non funzionali.

### Riassunto veloce

- stesse metriche in due spazi: `nt` e `bin`
- possibile filtro aggiuntivo: `_refbin`
- possibile estensione simmetrica: `_reciprocal`
- un file CSV per train e uno per validation, per ogni combinazione dei criteri

## Criteri di nomenclatura in `data/split_train_validation`

In questa cartella ci sono due strategie di split, ognuna con la sua sottocartella.

### 1) Split con VAE in validation + sequenze vicine

Cartella:

`split_VAE-in-validation_plus-close-to-them/`

Idea:

- il validation set contiene tutte le sequenze generate da VAE
- in piu contiene anche le sequenze non VAE ma vicine alle VAE (distanza di Hamming < 10)

Nomi file principali:

- `DTp_val_VAE-plus-close.fasta` / `DTm_val_VAE-plus-close.fasta`:
	validation con VAE + sequenze vicine alle VAE
- `DTp_VAE-close_at-dist-25-30.fasta` / `DTm_VAE-close_at-dist-25-30.fasta`:
	insieme delle sequenze VAE + close con vincolo esplicito di distanza 25-30 da Azoarcus nel nome
- `DTp_train_non-vae_minus-close-to-vae.fasta` / `DTm_train_non-vae_minus-close-to-vae.fasta`:
	training senza VAE e senza le sequenze troppo vicine alle VAE

### 2) Split random uniforme 15% + aggiunta insieme escluso

Cartella:

`split_fraction-0.15_ add-vae-25-30-to-validation/`

Idea:

- si parte da uno split random uniforme al 15%
- prima dello split si escludono le sequenze VAE a distanza 25-30 da Azoarcus
- dopo lo split, l'insieme escluso viene aggiunto al validation

Nomi file principali:

- `DTp_excluded-vae-at-dist-25-30.fasta` / `DTm_excluded-vae-at-dist-25-30.fasta`:
	insieme escluso inizialmente (VAE a distanza 25-30 da Azoarcus)
- `DTp_train_excluding-vae-25-30.fasta` / `DTm_train_excluding-vae-25-30.fasta`:
	training ottenuto dal random split, senza l'insieme escluso
- `DTp_val_plus-vae-25-30.fasta` / `DTm_val_plus-vae-25-30.fasta`:
	validation finale = validation random + insieme escluso (vae 25-30)

### Nota sui prefissi `DTp` e `DTm`

- `DTp` e `DTm` indicano due gruppi/dataset paralleli costruiti con la stessa logica di split
- i suffissi del nome file (`train`, `val`, `excluded`, `plus`, `minus`, `VAE-close`) descrivono il ruolo del file nello split

## Configurazione training (`src/dpo_config.py`)

Tutti gli input del training DPO sono centralizzati nella dataclass `Config` in `src/dpo_config.py`.

Nel codice (`src/dpo_train.py`) i path vengono risolti rispetto alla root del progetto con `resolve_path(...)`.
In pratica, se metti un path relativo (es. `data/...`), viene interpretato dalla cartella principale del repository.

### 1) Percorsi file (parte piu importante)

#### Path base e checkpoint iniziale

- `dn_path`: FASTA usato per costruire vocabolario/tokenizer e per metriche DN.
- `pretrained_ckpt_path`: checkpoint GPT di partenza da cui inizia il fine-tuning DPO/ReInt.

#### Dataset train/validation per DPO

- `train_good_fasta_path`: FASTA delle sequenze preferite (funzionali) del train.
- `train_bad_fasta_path`: FASTA delle sequenze non preferite (non funzionali) del train.
- `train_csv_mapping_path`: CSV pairing train; ogni riga (una good) contiene gli indici delle bad accoppiate.
- `val_good_fasta_path`: FASTA delle sequenze preferite per validation DPO.
- `val_bad_fasta_path`: FASTA delle sequenze non preferite per validation DPO.
- `val_csv_mapping_path`: CSV pairing validation con la stessa logica del train.

Vincolo importante letto dal codice (`src/dpo_data.py`):
- il numero di righe del CSV deve essere uguale al numero di sequenze in `*_good_fasta_path`.
- se non coincide, il loader fallisce con assertion.

#### Dataset opzionali extra (`val_1`, `val_2`)

- `val_1_good_fasta_path`, `val_1_bad_fasta_path`: dataset opzionale usato per metriche/plot di separazione (non per la loss DPO principale).
- `val_2_good_fasta_path`, `val_2_bad_fasta_path`: secondo dataset opzionale per analisi aggiuntive.
- se un path e `None` o stringa vuota, quel blocco viene disattivato automaticamente.

#### Path output

- `checkpoint_dir`: directory dove salva i checkpoint `dpo_model_epochX.pt`.
- `image_dir`: directory dove salva i plot per epoca.
- `history_json_path`: file JSON con la storia delle metriche.

Output extra generato dal codice:
- a fine training viene scritto anche `config_used.json` nella stessa cartella di `history_json_path`.

### 2) Iperparametri del modello

- `n_embd`: dimensione embedding/model width.
- `n_head`: numero di attention heads.
- `n_layer`: numero di blocchi transformer.

Questi tre parametri devono essere coerenti col checkpoint in `pretrained_ckpt_path` (stessa architettura), altrimenti il load dello state dict puo fallire.

Altri parametri legati alla sequenza:
- `block_size`: lunghezza usata per encoding/padding nei dataset DPO.
- `max_context`: contesto usato nel caricamento dataset DN (`load_dataset(...)`).

In questo progetto i default sono allineati (`198`), ed e consigliato mantenerli coerenti.

### 3) Iperparametri DPO e ReInt

- `reint`:
	`False` = usa loss DPO standard;
	`True` = usa loss ReInt.
- `beta`: scala del reward nelle loss (piu alto = differenze winner/loser piu pesate).
- `lambda_reint`: usato solo quando `reint=True`; pesa il termine del loser nella ReInt.

Comportamento nel codice (`src/dpo_metrics.py`):
- DPO usa modello policy e `ref_model` congelato per confronto relativo.
- ReInt attuale usa reward basato su log-prob del modello policy (i termini reference nella funzione sono commentati).

### 4) Freezing dei layer

Parametro chiave:
- `layers_to_freeze`: quanti blocchi transformer iniziali congelare.

Semantica precisa in `src/dpo_train.py`:
- se `layers_to_freeze > 0`, viene congelato anche l'embedding layer (`token_emb`).
- poi vengono congelati i primi `layers_to_freeze` blocchi (`model.layers[:layers_to_freeze]`).
- i parametri congelati non vengono passati all'optimizer (`filter(lambda p: p.requires_grad, model.parameters())`).

Quindi:
- `0`: nessun freezing.
- `N > 0`: embedding + primi `N` blocchi fissi, il resto si aggiorna.

### 5) Iperparametri training ed evaluation utili

- `batch_size`, `learning_rate`, `num_epochs`: training loop principale.
- `eval_every_n_epochs`: ogni quante epoche fare valutazione completa e salvataggio plot.
- `scheduler_step_size`, `scheduler_gamma`: scheduler `StepLR`.
- `use_torch_compile`: abilita `torch.compile` per model e ref_model (con fallback automatico se fallisce).
- `compute_full_nll_metrics`: se `True` calcola NLL su tutto il dataset (piu lento).
- `nll_rand_n_batches`, `nll_rand_eval_batch_size_cap`: stima NLL random-batch (piu veloce).
- `compute_val_separation_metric`: abilita metriche di separazione/correlazione per val, val_1, val_2.
- `dn_likelihood_mode`: `full` o `fixed_subsample` per monitoraggio likelihood su DN.
- `dn_fixed_subsample_size`, `dn_fixed_subsample_seed`: controllo del sottoinsieme DN fisso.

### 6) Checklist pratica prima di lanciare un run

- verifica che tutti i path in `Config` puntino a file esistenti.
- verifica coerenza tra FASTA good e righe del CSV pairing.
- verifica che `n_embd/n_head/n_layer` siano compatibili col checkpoint pretrained.
- scegli `reint`, `beta`, `lambda_reint` in modo esplicito prima del run.
- imposta `layers_to_freeze` in base al grado di fine-tuning desiderato.