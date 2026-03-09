# 2. RICERCA TECNICA
## Sviluppo di una Pipeline di Analisi Quantitativa Spazio-Timbrica

### 2.1 Motivazione e obiettivi

L'obiettivo della ricerca tecnica è sviluppare una metodologia scientifica per analizzare quantitativamente l'uso dello spazio nella composizione musicale per media immersivi. La domanda di ricerca centrale è: **come possiamo misurare oggettivamente il rapporto tra caratteristiche timbriche e configurazioni spaziali in contenuti audio object-based?**

Questa esigenza nasce da un gap metodologico identificato nella letteratura: mentre esistono studi approfonditi sulla percezione spaziale (Blauert, 1997; Rumsey, 2001) e sull'analisi del sound design cinematografico (Chion, 1994), manca un approccio quantitativo che correli sistematicamente le caratteristiche spettrali con i parametri spaziali per supportare decisioni compositive evidence-based. La pipeline sviluppata nel primo anno costituisce il **primo framework sistematico** per questo tipo di analisi integrata, e rappresenta lo strumento fondamentale per la validazione quantitativa della teoria TSF (Timbro-Spazio-Forma).

Il collegamento con la ricerca artistica è diretto: i dati estratti dalla pipeline vengono utilizzati per auto-analizzare le composizioni proprie (come *Book of Water*), creando un ciclo di **feedback quantitativo** che informa e valida le scelte compositive. Parallelamente, l'analisi di repertorio cinematografico esistente consente di identificare pattern ricorrenti nelle pratiche professionali, contribuendo alla costruzione della letteratura compositiva immersiva.

---

### 2.2 Il formato ADM: fondamento metodologico

La metodologia si basa sull'**Audio Definition Model (ADM)**, standard internazionale pubblicato originariamente come EBU Tech 3364 (2014) e successivamente adottato come ITU-R BS.2076 (2017). L'ADM fornisce un modello di metadati formalizzato per descrivere il contenuto e il formato di tracce audio in file destinati a esperienze immersive (EBU, 2018; ITU-R, 2017).

A differenza dei formati multicanale tradizionali (come il 5.1 surround), dove la posizione spaziale è dedotta implicitamente dalla differenza di ampiezza tra i canali, i protocolli **object-based** come il Dolby Atmos associano a ogni oggetto sonoro metadati espliciti che ne descrivono posizione tridimensionale, movimento, guadagno e dispersione (spread). Questa differenza è fondamentale per la ricerca: i metadati ADM consentono un'analisi quantitativa **precisa e riproducibile** dei parametri spaziali compositivi, aprendo possibilità nel dominio della data analysis che non erano disponibili con i formati precedenti.

La struttura ADM si articola in elementi gerarchici interconnessi:

| Elemento ADM | Descrizione | Informazioni estratte |
|--------------|-------------|----------------------|
| **audioObject** | Rappresenta un oggetto sonoro logico | Identificativo, timing, relazioni |
| **audioChannelFormat** | Formato di un singolo canale audio | Tipo di contenuto, riferimenti |
| **audioBlockFormat** | Suddivisione temporale del canale | Posizione (x, y, z), gain, spread |
| **audioPackFormat** | Combinazione di canali correlati | Configurazione spaziale complessiva |

Questa struttura permette di tracciare con precisione millimetrica la posizione e il movimento di ogni oggetto sonoro nel tempo, fornendo dati oggettivi per l'analisi compositiva.

---

### 2.3 Architettura della pipeline di analisi

Nel corso del primo anno è stata sviluppata una **pipeline software in Python** per l'estrazione e caratterizzazione simultanea di parametri spaziali e timbrici da contenuti audio ADM/Dolby Atmos. L'architettura si articola in quattro moduli principali:

**Modulo 1 - Parsing ADM**: Il primo stadio estrae i metadati spaziali dal chunk AXML contenuto nei file WAV BWF (Broadcast Wave Format). L'algoritmo interpreta la struttura XML dell'ADM, identificando audioObject, audioChannelFormat e audioBlockFormat, e ricostruisce le traiettorie spaziali di ciascun oggetto nel tempo. La normalizzazione avviene su una griglia temporale regolare (hop_size configurabile, default 10ms) per garantire sincronizzazione con l'analisi audio.

**Modulo 2 - Analisi spettrale**: Per ogni oggetto ADM, vengono estratte features timbriche utilizzando la libreria librosa (McFee et al., 2015), standard de facto per l'analisi audio in Python. Le finestre di analisi spettrale sono allineate temporalmente alla griglia dei metadati spaziali, garantendo correlazione precisa tra parametri timbrici e spaziali.

**Modulo 3 - Calcolo di features derivate**: Dalle traiettorie spaziali vengono calcolate metriche derivate (velocità, accelerazione, entropia spaziale) che caratterizzano il comportamento dinamico degli oggetti. L'entropia spaziale, in particolare, misura la distribuzione probabilistica della posizione su zone discrete dello spazio, fornendo un indicatore di "dispersione" o "concentrazione" del contenuto spaziale.

**Modulo 4 - Correlazione e indicizzazione**: Il modulo finale calcola correlazioni statistiche tra features spaziali e spettrali, e genera indici compositi per la caratterizzazione compositiva.

---

### 2.4 Features estratte: dominio spettrale e spaziale

L'innovazione metodologica principale consiste nell'**integrazione sincronizzata** di due domini di analisi tradizionalmente separati. Di seguito le features implementate:

#### Features spettrali (dominio timbrico)

| Feature | Descrizione | Rilevanza compositiva |
|---------|-------------|----------------------|
| **MFCC** (13 coefficienti) | Mel-Frequency Cepstral Coefficients, rappresentano la forma spettrale secondo la scala percettiva Mel | Caratterizzazione timbrica compatta, correlata alla percezione uditiva umana (Barua et al., 2024) |
| **Centroide spettrale** | "Centro di massa" dello spettro, indica la frequenza media pesata | Misura di "brillantezza" timbrica |
| **Bandwidth spettrale** | Ampiezza della distribuzione spettrale | Indica "apertura" o "chiusura" del timbro |
| **Contrasto spettrale** | Differenza tra picchi e valli nello spettro | Distingue suoni armonici da rumorosi |
| **Rolloff spettrale** | Frequenza sotto cui si concentra il 85% dell'energia | Caratterizza la "pesantezza" timbrica |
| **RMS energy** | Energia media del segnale | Dinamica e intensità percepita |

Gli MFCC (Mel-Frequency Cepstral Coefficients) meritano particolare attenzione: questa rappresentazione emula il sistema uditivo umano attraverso la scala Mel, che comprime le frequenze alte e espande quelle basse in modo analogo alla coclea. Studi recenti confermano che gli MFCC producono features altamente distintive per la classificazione audio (Barua et al., 2024; Joysingh, 2025).

#### Features spaziali (dominio posizionale)

| Feature | Descrizione | Formula/Metodo |
|---------|-------------|----------------|
| **Posizione cartesiana** | Coordinate x (left-right), y (front-back), z (height) | Mapping diretto da ADM |
| **Velocità 3D** | Tasso di variazione della posizione | Derivata numerica delle coordinate |
| **Accelerazione 3D** | Tasso di variazione della velocità | Derivata seconda |
| **Entropia spaziale** | Misura di dispersione della distribuzione spaziale | \( H = -\sum p_i \log(p_i) \) su zone discrete |
| **Spread/Width/Depth** | Parametri di dispersione dell'oggetto ADM | Estratti direttamente dai metadati |
| **Classificazione zonale** | Categorizzazione in regioni (front/rear, left/right, high/low) | Soglie configurabili |

---

### 2.5 Validazione: confronto Object-Based vs Channel-Based

Un primo obiettivo sperimentale è stato validare la superiorità metodologica dell'analisi object-based rispetto a quella channel-based. Sono stati condotti test comparativi esportando la stessa sorgente audio con identico movimento spaziale in due formati: come **Object** (con metadati ADM) e come **Bed** (formato multicanale tradizionale, dove il movimento è dedotto dalla differenza di ampiezza tra canali).

**Test 1 - Segnale sinusoidale con movimento lineare**: Una sinusoide con movimento centrale dal back al front (10 secondi) è stata analizzata in entrambi i formati. L'analisi object-based ha prodotto una traiettoria lineare e precisa, mentre l'analisi channel-based, pur correlata (r > 0.85), ha presentato maggiore rumore e deviazioni dalla linearità attesa.

**Test 2 - Due sorgenti sinusoidali simultanee**: Con l'introduzione di una seconda sorgente, l'analisi channel-based ha mostrato difficoltà nella separazione dei movimenti, mentre l'approccio object-based ha mantenuto tracciamento indipendente per ciascun oggetto.

**Test 3 - Sorgenti con spettro complesso**: Utilizzando materiale audio realistico (voce, strumenti), l'analisi object-based ha mantenuto robustezza, mentre quella channel-based ha degradato significativamente, confondendo variazioni timbriche con variazioni spaziali.

Questi risultati confermano che **l'ADM è la metodologia preferibile per analisi quantitative rigorose**, pur validando approcci alternativi per repertorio pre-Atmos dove i metadati non sono disponibili (cfr. Mouchtaris et al., 2021).

---

### 2.6 Proposta di un Indice di Immersività

Sulla base delle correlazioni spazio-spettrali identificate, è stato sviluppato un **Indice di Immersività (I_imm)** come misura composita del coinvolgimento percettivo potenziale di un contenuto audio spaziale:

\[
I_{imm} = \frac{H_{spatial} \cdot \sigma_{spectral} \cdot D_{movement}}{N}
\]

Dove:
- \( H_{spatial} \) = entropia spaziale (distribuzione della posizione)
- \( \sigma_{spectral} \) = varianza spettrale (variabilità timbrica)
- \( D_{movement} \) = densità di movimento (attività motoria nel tempo)
- \( N \) = fattore di normalizzazione

L'indice si basa sull'ipotesi, supportata dalla letteratura sulla percezione immersiva (Kern & Ellermeier, 2020; Zhang et al., 2023), che l'immersività percepita sia correlata positivamente con: (a) la distribuzione spaziale del contenuto sonoro, (b) la varietà timbrica, e (c) l'attività dinamica degli oggetti. La validazione percettiva dell'indice attraverso listening tests controllati è prevista per il secondo anno.

---

### 2.7 Risultati preliminari e pattern identificati

La pipeline è stata applicata a un corpus di test comprendente:
- Segnali sintetici (validazione ground truth)
- Estratti cinematografici in formato Atmos (*Gravity*, *Dunkirk*, *Blade Runner 2049*)
- Composizioni proprie (*Book of Water*, work in progress)

Sono emersi **pattern significativi** che suggeriscono correlazioni ricorrenti tra scelte timbriche e spaziali:

| Contesto narrativo | Correlazione identificata | Significatività |
|-------------------|---------------------------|-----------------|
| Cinema d'azione | Velocità spaziale ↔ Centroide spettrale (r = 0.52) | p < 0.001 |
| Scene intime | Concentrazione in zona center-front (78% del tempo) | — |
| Transizioni narrative | Shift spaziale preceduto da variazione timbrica | Lead time ~1.2s |
| Suoni ad alta frequenza | Tendenza a localizzazione anteriore | r = 0.34, p < 0.001 |

Questi risultati confermano intuizioni della pratica professionale e suggeriscono che esistono **"grammatiche implicite"** nell'uso compositivo dello spazio cinematografico che possono essere codificate e studiate sistematicamente.

---

### 2.8 Strumenti di visualizzazione

È stata sviluppata un'applicazione interattiva in **Streamlit** per l'esplorazione e visualizzazione dei dati estratti, democratizzando l'accesso agli strumenti di ricerca TSF anche per utenti non specialisti in programmazione. L'interfaccia consente:

- **Timeline Analysis**: visualizzazione temporale di oggetti attivi, density maps front/rear
- **Spatial Mapping**: scatter plots 2D/3D delle traiettorie con color-coding temporale
- **Spectro-Spatial Correlation**: heatmaps di correlazione tra MFCC e coordinate spaziali
- **Per-Object Spectrograms**: spettrogrammi individuali con overlay di traiettorie
- **Statistical Testing**: t-tests automatici per confronti statistici

---

### 2.9 Collegamento con la ricerca artistica e la letteratura compositiva

La pipeline tecnica non è un fine in sé, ma uno **strumento al servizio della pratica compositiva e della teorizzazione estetica**. Il collegamento con le altre componenti della ricerca si articola su tre livelli:

**Auto-analisi delle composizioni proprie**: I file ADM delle composizioni in corso (come *Book of Water*) vengono analizzati con la pipeline per verificare se le intenzioni compositive si traducono in pattern spazio-timbrici misurabili. Questo crea un ciclo di feedback che informa la revisione del materiale.

**Validazione della teoria TSF**: I dati quantitativi forniscono evidenze empiriche per le ipotesi della teoria Timbro-Spazio-Forma. Ad esempio, la correlazione identificata tra variazioni timbriche e shift spaziali supporta l'idea che timbro e spazio siano parametri co-dipendenti nella percezione formale.

**Costruzione della letteratura compositiva**: L'analisi di repertorio cinematografico canonico consente di identificare pattern ricorrenti e "buone pratiche" che possono essere codificate in linee guida compositive evidence-based, contribuendo al corpus teorico per la composizione immersiva.

---

### 2.10 Limitazioni e sviluppi futuri

**Limitazioni identificate**:
- Sensibilità al rumore per movimenti micro-spaziali (< 0.01 units/s)
- Dipendenza dalla qualità dei metadati ADM (alcuni mix professionali presentano incongruenze)
- Necessità di calibrazione percettiva per validare gli indici compositivi

**Obiettivi per il secondo anno**:
- Implementazione di machine learning per classificazione automatica di gesti spazio-timbrici
- Estensione dell'analisi a film completi con segmentazione semantica
- Validazione percettiva dell'Indice di Immersività attraverso listening tests controllati
- Sviluppo di un prototipo di sistema compositivo assistito basato sui pattern identificati
- Pubblicazione di dataset annotato per la comunità scientifica

---

## Riferimenti bibliografici (Sezione 2)

Barua, N., et al. (2024). "Enhancing Audio Classification Through MFCC Feature Extraction and Data Augmentation." *International Journal of Advanced Computer Science and Applications*, 15(7).

Blauert, J. (1997). *Spatial Hearing: The Psychophysics of Human Sound Localization*. MIT Press.

Chion, M. (1994). *Audio-Vision: Sound on Screen*. Columbia University Press.

EBU (2018). Tech 3364 v2 - Audio Definition Model. European Broadcasting Union.

ITU-R (2017). Recommendation BS.2076-1 - Audio Definition Model. International Telecommunication Union.

Joysingh, S. J. (2025). "Significance of chirp MFCC as a feature in speech and audio applications." *Speech Communication*.

Kern, A. C., & Ellermeier, W. (2020). "Audio in VR: Effects of a Soundscape and Movement-Triggered Step Sounds on Presence." *Frontiers in Robotics and AI*, 7, 20.

McFee, B., et al. (2015). "librosa: Audio and music signal analysis in python." *Proceedings of the 14th Python in Science Conference*.

Mouchtaris, A., et al. (2021). "Immersive Audio Signal Processing and Rendering: From Objects to Experiences." *IEEE Signal Processing Magazine*, 38(3), 16-31.

Rumsey, F. (2001). *Spatial Audio*. Focal Press.

Zhang, Y., et al. (2023). "A survey of immersive visualization: Focus on perception and interaction." *Virtual Reality & Intelligent Hardware*.