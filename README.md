# Traffic Sign Detenction and Recognition with YOLOv8

**Nome del Progetto**: Traffic Sign Detenction and Recognition  

**Descrizione**: Progetto svolto come parte conclusiva dell’esame di Machine Learning e Sistemi Intelligenti per Internet del corso di laurea magistrale di Ingegneria Informatica tenuta all’Università di Roma Tre.
Lo scopo di questo lavoro è quello di realizzare un sistema di Object Detenction e Object Recognition per il rilevamento e la classificazione di segnaletica stradale verticale.
Questi due task fanno riferimento a un ambito dell’intelligenza artificiale chiamato Computer Vision, le cui tecniche permettono ai nostri sistemi di ricavare informazioni significative e rilevanti da immagini digitali, video e altri input visivi.
Queste tecniche sono estremamente pervasive, infatti posso essere applicate in molti campi, quali quello della robotica, della guida autonoma, dell’anticontraffazione, del marketing, della medicina diagnostica e della sicurezza.
Il progetto è stato svolto in modo cooperativo dagli studenti Paolo Tardioli e Alessandro Tibaldi. Il processo che ha portato al risultato finale non è stato un processo lineare, bensì ci siamo trovati frequentemente costretti a cambiare il nostro approccio e i modi di affrontare i problemi che abbiamo incontrato, dovendo alle volte ripartire da zero. Per questa ragione nella relazione verranno mostrati non solo dei test andati a buon fine, ma anche tutti quelli che non hanno portato ai risultati sperati, accompagnati dalle nostre considerazioni.

## Contenuti del Repository

Il repository include le seguenti directories:

**Notebooks**: Contiene gli script Python per generare i dataset per l'addestramento delle reti neurali di Detenction e Classification  
**Models**: Contiene tutti i modelli addestrati ottenuti dai vari test effettuati  
**Data**: Contiene i link ai dataset utilizzati caricati su Google Drive  
**Training**: Contiene i notebooks di Jupyter utilizzati per l'addestramento dei modelli  
**Predict**: Contiene i notebooks di Jupyter utilizzati per effettuare predizioni e per fare valutazioni sui modelli addestrati  
**Config**: COntiene tutti i file di configurazione  

## Dataset

I dataset per l'addestramento della rete di rilevamento è reperibile a questo link: https://drive.google.com/drive/folders/1s3aaxcOK5s2pqNI3irhyN-lUpgwZiZ1q?usp=drive_link  

Il dataset per l'addestramento della rete di classificazione è reperibile a questo link: https://drive.google.com/drive/folders/1zMN0YBkUlFeAwOkOJOUiMSdingsAFXTU?usp=drive_link

## File di Configurazione

[config_dir.yaml](config%2Fconfig_dir.yaml) : una volta scaricati di dataset accertarsi di modificare il file di configurazione delle directories in base alle proprie esigenze.  

[config_ds_detenct.yaml](config%2Fconfig_ds_detenct.yaml) : questo file di configurazione è relativo all'addestramento del modello di Detenction. Permette di settare alcuni parametri
come le proporzioni delle partizioni di train, test e validation, nonché la mappatura delle etichette.

[config_ds_classify.yaml](config%2Fconfig_ds_classify.yaml) : questo file di configurazione è relativo all'addestramento del modello di Classification. Permette di settare alcuni parametri
come le proporzioni delle partizioni di train e test, nonché la mappatura delle etichette.

## Generazione del dataset per l'addestramento del modello di Detenction

Una volta modificato adeguatamente i file di configurazione appena elencati, è sufficiente eseguire tutti i blocchi del modulo [generate_dataset.ipynb](notebooks%2Fdetenction%2Fgenerate_dataset.ipynb)  
Aspettando il completamento delle operazioni, verrà generato il dataset finale, pronto per l'addestramento con YOLOv8, nella cartella di destinazione specificata.

## Generazione del dataset per l'addestramento del modulo di Classification 

Una volta modificato adeguatamente i file di configurazione appena elencati, è sufficiente eseguire tutti i blocchi del modulo [generate_dataset.ipynb](notebooks%2Fclassification%2Fgenerate_dataset.ipynb)   
Aspettando il completamento delle operazioni, verrà generato il dataset finale, pronto per l'addestramento con YOLOv8, nella cartella di destinazione specificata.

## Modelli addestrati

I modelli addestrati sono presenti nella cartella models. Oltre ai modelli sono presenti tutti di dati relativi, raccolti durante i cicli di addestramento, matrice di confusione, andamento delle funzioni di loss, metriche etc..  
Tra i modelli di detenction troviamo:
<ul>
    <li>[train_7_classes_split_60_20_20](models%2Fdetenction%2Ftrain_7_classes_split_60_20_20)</li>
    <li>[train_7_classes_split_70_15_15](models%2Fdetenction%2Ftrain_7_classes_split_70_15_15)</li>
    <li>[train_7_classes_split_80_10_10](models%2Fdetenction%2Ftrain_7_classes_split_80_10_10)</li>
    <li>[train_46_classes_split_70_15_15](models%2Fdetenction%2Ftrain_46_classes_split_70_15_15)</li>
</ul>
Tra i modelli di classification troviamo:
<ul>
    <li>[traint_43_classes_split_70_30](models%2Fclassification%2Ftraint_43_classes_split_70_30)</li>
</ul>

