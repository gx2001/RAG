# R.A.G 

## Στοιχεία Ομάδας
- **Stathis Andreopoulos** - ΑΜ: 4630
- **Giorgos Hatziligos** - ΑΜ: 4835

###Εκτέλεση Έφαρμογης R.A.G
1.άνοιγμα του αρχείου rag_anaktisi_step_by_step.ipynb με κάποιο ide  π.χ vscode
2.εγκατάσταση των παρακάτω dependencies
3.εγκατασταση dataset 
4.εκτελεστε την εντολη run all 


### Εγκατάσταση Dependencies
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity


### Εγκατάσταση και Διαμόρφωση Ollama
1. Κατεβάστε και εγκαταστήστε το Ollama από: https://ollama.com/download
2. Εκτελέστε: `ollama pull llama3.2:1b`
3. Εκκινήστε την υπηρεσία: `ollama serve`


## Αναλυτική Περιγραφή Αρχιτεκτονικής

###  Φόρτωση και Αρχική Επεξεργασία Δεδομένων (Step 1)

**Dataset Χαρακτηριστικά**: Χρησιμοποιούμε το dataset `CNN_Articles_clean.csv` που περιέχει **4,076 ειδησεογραφικά άρθρα** από το CNN, καλύπτοντας ένα ευρύ φάσμα θεμάτων όπως πολιτική, οικονομία, αθλητικά, τεχνολογία και διεθνή νέα. Κάθε άρθρο περιλαμβάνει structured metadata όπως συγγραφέα, ημερομηνία δημοσίευσης, κατηγορία, URL και φυσικά το πλήρες κείμενο.

**Προεπεξεργασία Δεδομένων**: Στο step αυτό φορτώνουμε τα δεδομένα μέσω της βιβλιοθήκης pandas και εκτελούμε βασικό data cleaning. Ελέγχουμε για missing values στα κρίσιμα πεδία (Headline, Article text) και εξασφαλίζουμε την ακεραιότητα των δεδομένων πριν προχωρήσουμε στο chunking.

### Τεχνολογία Chunking και Διαμερισμός Κειμένου (Step 2)

**Μεθοδολογία Chunking**: Υλοποιούμε την τεχνική `RecursiveCharacterTextSplitter` από το LangChain framework. Η μέθοδος αυτή είναι ιδιαίτερα αποτελεσματική γιατί προσπαθεί να διατηρήσει την σημασιολογική συνοχή του κειμένου.

**Τεχνικές Παράμετροι Chunking**:
- **Chunk Size**: 1,000 χαρακτήρες - επιλέχθηκε ως optimal balance μεταξύ του να έχουμε αρκετό context και του να μην έχουμε υπερβολικά μεγάλα chunks που θα μπορούσαν να περιέχουν άσχετες πληροφορίες
- **Overlap**: 200 χαρακτήρες - εξασφαλίζει ότι σημαντικές πληροφορίες που βρίσκονται στα όρια των chunks δεν θα χαθούν
- **Separators**: Ιεραρχικοί διαχωριστές `["\n\n", "\n", ".", " ", ""]` που προτεραιοποιούν φυσικά σημεία διακοπής του κειμένου

**Δομή Κειμένου**: Για κάθε άρθρο, συνδυάζουμε τα πεδία `Headline` και `Article text` 

Αυτή η προσέγγιση εξασφαλίζει ότι κάθε chunk περιέχει τον τίτλο του άρθρου για καλύτερο context και διευκολύνει την κατανόηση του περιεχομένου.

**Metadata Management**: Κάθε chunk εμπλουτίζεται με structured metadata που περιλαμβάνει:
- `title`: Πλήρης τίτλος άρθρου για context
- `article_id`: Unique identifier του άρθρου στο dataset
- `chunk_id`: Sequential ID του chunk εντός του άρθρου
- `source`: Σταθερή τιμή "CNN" για source attribution


### Σύστημα Embeddings και Σημασιολογική Αναπαράσταση (Step 3)

**Επιλογή Embedding Model**: Χρησιμοποιούμε το `sentence-transformers/all-mpnet-base-v2` model από την HuggingFace, το οποίο αποτελεί έναν από τους πιο αποτελεσματικούς sentence transformers για semantic similarity tasks. Το μοντέλο αυτό έχει προπονηθεί σε μεγάλα datasets και παρέχει high-quality embeddings με 768 διαστάσεις.

**Τεχνικά Χαρακτηριστικά Embeddings**:
- **Model Architecture**: MPNet (Masked and Permuted Pre-training for Language Understanding)
- **Embedding Dimensions**: 768
- **Normalization**: True - εξασφαλίζει ότι όλα τα embeddings έχουν unit length για accurate cosine similarity
- **Device Configuration**: CPU - για compatibility και ease of deployment

**Αιτιολόγηση Επιλογής**: Το all-mpnet-base-v2 επιλέχθηκε έναντι άλλων alternatives όπως το all-MiniLM-L6-v2 (ταχύτερο αλλά χαμηλότερη ποιότητα) ή το all-mpnet-large (υψηλότερη ποιότητα αλλά significantly slower) γιατί προσφέρει το optimal trade-off μεταξύ ποιότητας και performance.

### Vector Database και Indexing (Step 3.1)

**FAISS Implementation**: Για την αποθήκευση και αναζήτηση των embeddings χρησιμοποιούμε το Facebook AI Similarity Search (FAISS) library.

**Database Configuration**:
- **Index Type**: Flat index για exact search 
- **Similarity Metric**: Cosine similarity 
- **Storage Strategy**: storage στο filesystem (`faiss_index/`) 
- **Memory Management**: Optimized για CPU operations

**Δημιουργία Index**: Το FAISS index δημιουργείται μέσω της `FAISS.from_texts()` μεθόδου που παίρνει ως input τα text chunks, το embedding model και τα metadata. Η διαδικασία αυτή εκτελείται μια φορά και το index αποθηκεύεται για μελλοντική χρήση.

### Large Language Model- LLM(Step 4)

**LLM Setup**: Χρησιμοποιούμε το Llama 3.2:1b model μέσω του Ollama framework. Το Llama 3.2:1b επιλέχθηκε ως compromise μεταξύ performance και resource requirements, προσφέροντας καλή ποιότητα text generation με reasonable computational overhead.Επισης , χρησιμοποιήσαμε και σε αλλα πειραματα το Llama 3.2:3b model μέσω του Ollama framework. 

**Παράμετροι Μοντέλου**:
- **Temperature**: 0.0 - εξασφαλίζει deterministic και 0.1 ωστε να παραγει καλυτερα αποτελεσματα ωστε να συμμετεχει και το llm
- **Local Deployment**: Μέσω Ollama server 
- **Model Size**: 1 billion parameters - επαρκές για news domain tasks ollama3.2:.1b και ollama3.2:1¨3b με 3 billion parameters

**Testing και Validation**: Εκτελούμε validation test με test query για να επιβεβαιώσουμε ότι το LLM λειτουργεί σωστά πριν την ενσωμάτωση στο RAG pipeline.

### RAG Pipeline (Step 5)

**LangChain RetrievalQA**: Το κεντρικό component του συστήματός μας είναι η LangChain RetrievalQA chain που ενσωματώνει όλα τα προηγούμενα components σε ένα cohesive pipeline.

**Chain Configuration**:
- **Chain Type**: "stuff" strategy - concatenates όλα τα retrieved documents στο prompt
- **Retriever**: FAISS vector store as retriever με top-k search
- **K Parameter**: 20 - retrieve τα 20 πιο σχετικά chunks για κάθε query μετα χρησιμοποιήσαμε 10 - retrieve τα 10 πιο σχετικά chunks

**Prompt**: Σχεδιάσαμε ένα specialized prompt template που:
- Καθορίζει το ρόλο του assistant ως news expert
- Δίνει explicit instructions για χρήση μόνο των provided sources
- Εξασφαλίζει structured output format

### Σύστημα Αξιολόγησης και Similarity Analysis (Step 5-Functions)

**Query Processing Functions**: Αναπτύξαμε δύο κύριες functions για comprehensive evaluation:

**`query_with_rag()`**: Εκτελεί complete RAG pipeline και επιστρέφει:
- Generated answer από το LLM
- Source documents που χρησιμοποιήθηκαν
- Detailed similarity scores για transparency

**`query_without_rag()`**: Εκτελεί direct LLM query χωρίς retrieval για comparison purposes.

**Similarity Score Calculation**: Η `get_retrieval_similarity_scores()` function υπολογίζει cosine similarity μεταξύ query embedding και retrieved document embeddings, παρέχοντας:
- Individual similarity scores για κάθε retrieved chunk
- Statistical analysis (mean, max, min)
- Ranking των chunks κατά relevance

### Comprehensive Evaluation Framework (Step 6)

**Test Query Design**: Σχεδιάσαμε 5 representative queries που καλύπτουν διαφορετικές κατηγορίες και complexity levels:


**Evaluation Metrics**: Για κάθε query καταγράφουμε :
- Response quality comparison (με RAG vs χωρίς RAG)
- Retrieval effectiveness (similarity scores)


**Output Management**: Όλα τα αποτελέσματα αποθηκεύονται σε structured format (`rag_evaluation_results_with_similarity.csv`) για detailed analysis και reporting.



3. **System Initialization**: Εκτελέστε το notebook cell-by-cell για να παρακολουθήσετε τη διαδικασία:
   - Data loading 
   - Chunking
   - Embedding model initialization  
   - Vector database creation
   - RAG pipeline configuration
   - LLM setup και testing
   - Evaluation 
   
