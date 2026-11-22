import sys
import numpy as np
import faiss
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import z3
from sentence_transformers import SentenceTransformer
import spacy
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QTextEdit, QLabel, QGroupBox, QSplitter, QComboBox,
    QSpinBox, QPlainTextEdit, QFormLayout, QLineEdit
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RAGNeuroSymbolicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Symbolic RAG: Prompt-Driven Logic Generation")
        self.resize(1400, 900)
        
        self.faiss_index = None
        self.vectors = None
        self.metadata = []
        self.pos_tags = []  
        self.constraints = []
        self.knn_edges = []
        self.query_vector = None
        self.z3_opt = z3.Optimize()
        
        self.log("Loading AI Models...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 1. Knowledge Base (Vector DB)
        group_db = QGroupBox("1. Knowledge Base (Document Store)")
        db_layout = QVBoxLayout()
        db_layout.addWidget(QLabel("Load a large corpus of text:"))
        self.text_input = QPlainTextEdit()
        self.text_input.setPlainText(
            "The cat sat on the mat. A dog ran fast. The quick fox jumps over the lazy hound. "
            "A feline rested on the rug. The canine sprinted. A swift fox leaps above the tired dog. "
            "The sky was blue. The sun was bright. The bird flew high. The fish swam deep."
        )
        self.text_input.setMaximumHeight(100)
        db_layout.addWidget(self.text_input)
        
        self.btn_load_db = QPushButton("Parse Corpus & Build Vector Database")
        self.btn_load_db.clicked.connect(self.load_data)
        db_layout.addWidget(self.btn_load_db)
        group_db.setLayout(db_layout)
        left_layout.addWidget(group_db)

        # 2. RAG Prompt
        group_rag = QGroupBox("2. Semantic Query (RAG)")
        rag_layout = QFormLayout()
        self.query_input = QLineEdit("What did the cat do?")
        rag_layout.addRow("Prompt:", self.query_input)
        
        self.btn_retrieve = QPushButton("Retrieve Relevant Vectors")
        self.btn_retrieve.clicked.connect(self.retrieve_vectors)
        rag_layout.addRow(self.btn_retrieve)
        
        group_rag.setLayout(rag_layout)
        left_layout.addWidget(group_rag)

        # 3. Spectral Analysis
        group_spectral = QGroupBox("3. Sparse Spectral Analysis (Local Fluency)")
        spectral_layout = QVBoxLayout()
        self.knn_spinbox = QSpinBox()
        self.knn_spinbox.setMinimum(1)
        self.knn_spinbox.setMaximum(5)
        self.knn_spinbox.setValue(2)
        spectral_layout.addWidget(QLabel("k-Nearest Neighbors (k):"))
        spectral_layout.addWidget(self.knn_spinbox)
        
        self.btn_spectral = QPushButton("Compute Semantic Graph")
        self.btn_spectral.clicked.connect(self.compute_sparse_spectral)
        spectral_layout.addWidget(self.btn_spectral)
        group_spectral.setLayout(spectral_layout)
        left_layout.addWidget(group_spectral)

        # 4. Syntactic Template
        group_template = QGroupBox("4. Grammar Template (Z3 Hard Logic)")
        template_layout = QFormLayout()
        self.template_input = QLineEdit("DT, NN, VBD, IN, DT, NN")
        template_layout.addRow("POS Pattern:", self.template_input)
        group_template.setLayout(template_layout)
        left_layout.addWidget(group_template)

        # 5. Z3 Optimizer
        group_smt = QGroupBox("5. Z3 Dual-Objective Optimizer")
        smt_layout = QVBoxLayout()
        self.btn_smt = QPushButton("Generate Sentence (Relevance + Fluency)")
        self.btn_smt.clicked.connect(self.run_z3_optimizer)
        smt_layout.addWidget(self.btn_smt)
        group_smt.setLayout(smt_layout)
        left_layout.addWidget(group_smt)

        # 6. Generative Output & XAI
        group_xai = QGroupBox("6. Generative Output & XAI")
        xai_layout = QVBoxLayout()
        
        xai_layout.addWidget(QLabel("Final Generated Sentence:"))
        self.sentence_output = QLineEdit()
        self.sentence_output.setReadOnly(True)
        self.sentence_output.setStyleSheet("font-size: 16pt; font-weight: bold;")
        xai_layout.addWidget(self.sentence_output)
        
        self.xai_text = QTextEdit()
        self.xai_text.setReadOnly(True)
        xai_layout.addWidget(self.xai_text)
        group_xai.setLayout(xai_layout)
        left_layout.addWidget(group_xai)

        # Log
        group_log = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        group_log.setLayout(log_layout)
        left_layout.addWidget(group_log)

        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])

    def log(self, message):
        self.log_text.append(message)

    def load_data(self):
        self.log_text.clear()
        text = self.text_input.toPlainText()
        doc = self.nlp(text)
        
        words = []
        pos_tags = []
        
        seen = set()
        for token in doc:
            if token.text not in seen and not token.is_punct and not token.is_stop:
                seen.add(token.text)
                words.append(token.text)
                pos_tags.append(token.pos_)
                
        if len(words) < 2:
            self.log("Error: Please enter a longer corpus.")
            return
            
        self.metadata = words
        self.pos_tags = pos_tags
        
        self.log("Parsed Knowledge Base:")
        for w, p in zip(self.metadata, self.pos_tags):
            self.log(f"  {w} ({p})")
            
        self.vectors = self.embedder.encode(words).astype('float32')
        
        dim = self.vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self.vectors)
        self.log(f"\nVector Database built with {self.faiss_index.ntotal} concepts.")
        self.knn_edges = []
        self.plot_sequence("Semantic Knowledge Base", [])

    def retrieve_vectors(self):
        if self.faiss_index is None:
            self.log("Error: Load data first!")
            return
            
        query = self.query_input.text()
        self.log(f"\n--- RAG Phase ---")
        self.log(f"Embedding prompt: '{query}'")
        
        # Embed the query
        self.query_vector = self.embedder.encode([query]).astype('float32')
        
        # Retrieve top 15 most relevant words
        D, I = self.faiss_index.search(self.query_vector, 15)
        retrieved_indices = I[0]
        
        # Filter to only include retrieved words
        self.metadata = [self.metadata[i] for i in retrieved_indices]
        self.pos_tags = [self.pos_tags[i] for i in retrieved_indices]
        self.vectors = self.vectors[retrieved_indices]
        
        # Rebuild FAISS with only retrieved vectors
        self.faiss_index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.faiss_index.add(self.vectors)
        
        self.log("Retrieved Concepts (Top 15):")
        for w, p in zip(self.metadata, self.pos_tags):
            self.log(f"  {w} ({p})")
            
        self.knn_edges = []
        self.plot_sequence("Retrieved Vectors (RAG)", [])

    def compute_sparse_spectral(self):
        if self.faiss_index is None:
            self.log("Error: Retrieve vectors first!")
            return
            
        self.log("\n--- Spectral Phase ---")
        n = self.vectors.shape[0]
        k = self.knn_spinbox.value()
        
        D, I = self.faiss_index.search(self.vectors, k)
        self.knn_edges = []
        self.constraints = []
        
        row_ind, col_ind, data = [], [], []
        for i in range(n):
            for j in range(k):
                neighbor_idx = I[i][j]
                if neighbor_idx != i:
                    row_ind.append(i)
                    col_ind.append(neighbor_idx)
                    weight = 1.0 / (1.0 + D[i][j])
                    data.append(weight)
                    self.knn_edges.append((i, neighbor_idx))
                    self.constraints.append({'pts': (i, neighbor_idx), 'weight': weight})
                    
        self.log(f"Generated {len(self.constraints)} weighted soft edges for local fluency.")
        self.plot_sequence("Local Semantic Graph", [])

    def run_z3_optimizer(self):
        if not self.constraints:
            self.log("Error: Compute spectral constraints first!")
            return
            
        self.log("\n--- Z3 RAG Generation Phase ---")
        self.sentence_output.clear()
        self.xai_text.clear()
        self.z3_opt = z3.Optimize()
        
        n = len(self.vectors)
        template_str = self.template_input.text()
        template_pos = [t.strip() for t in template_str.split(',')]
        seq_len = len(template_pos)
        
        # 1. Hard Constraint: Exactly 'seq_len' words
        Include = z3.Function('Include', z3.IntSort(), z3.BoolSort())
        total_included = z3.Sum([z3.If(Include(i), 1, 0) for i in range(n)])
        self.z3_opt.add(total_included == seq_len)
        
        Pos = z3.Function('Pos', z3.IntSort(), z3.IntSort())
        
        for i in range(n):
            self.z3_opt.add(z3.Implies(Include(i), z3.And(Pos(i) >= 0, Pos(i) < seq_len)))
            for j in range(i+1, n):
                self.z3_opt.add(z3.Implies(z3.And(Include(i), Include(j)), Pos(i) != Pos(j)))

        # 2. Hard Logic: Template Fitting
        for slot_idx, required_pos in enumerate(template_pos):
            valid_indices = [i for i, tag in enumerate(self.pos_tags) if tag == required_pos]
            if not valid_indices:
                self.log(f"Error: No retrieved words match POS '{required_pos}'")
                return
                
            slot_constraints = []
            for i in valid_indices:
                self.z3_opt.add(z3.Implies(Pos(i) == slot_idx, Include(i)))
                slot_constraints.append(Pos(i) == slot_idx)
            self.z3_opt.add(z3.Or(slot_constraints))

        # 3. Soft Constraint A: Local Fluency (Maximize adjacency similarity)
        fluency_reward = z3.IntVal(0)
        for c in self.constraints:
            i, j = c['pts']
            weight = int(c['weight'] * 100)
            both_included = z3.And(Include(i), Include(j))
            dist = z3.If(Pos(i) >= Pos(j), Pos(i) - Pos(j), Pos(j) - Pos(i))
            is_adjacent = (dist == 1)
            reward = z3.If(z3.And(both_included, is_adjacent), weight, 0)
            fluency_reward = fluency_reward + reward
            
        # 4. Soft Constraint B: Global Relevance (Maximize similarity to Prompt)
        relevance_reward = z3.IntVal(0)
        query_sims = cosine_similarity(self.vectors, self.query_vector)[0]
        
        for i in range(n):
            # Scale similarity to integer weight
            q_weight = int(query_sims[i] * 1000)
            # If included, add its relevance to the prompt
            relevance_reward = relevance_reward + z3.If(Include(i), q_weight, 0)
            
        # Z3 Optimization: Maximize both (Fluency + Relevance)
        # We weight Relevance higher to ensure it answers the prompt
        self.z3_opt.maximize(fluency_reward + (relevance_reward * 10))
        
        self.log(f"Generating sentence for template: {template_pos}...")
        result = self.z3_opt.check()
        
        if result == z3.sat:
            self.log("✅ Z3 Result: OPTIMAL RAG SENTENCE GENERATED")
            model = self.z3_opt.model()
            
            sequence = []
            for i in range(n):
                if model.evaluate(Include(i)):
                    pos_val = model.evaluate(Pos(i)).as_long()
                    sequence.append((pos_val, i))
                    
            sequence.sort()
            ordered_indices = [idx for pos, idx in sequence]
            ordered_words = [self.metadata[idx] for idx in ordered_indices]
            
            final_sentence = " ".join(ordered_words) + "."
            self.sentence_output.setText(final_sentence)
            
            explanation = "✅ Explanation of Neuro-Symbolic RAG:\n\n"
            explanation += "1. Neural Retrieval (RAG): Embedded your prompt and retrieved the top 15 most semantically relevant words from the database.\n"
            explanation += "2. Semantic Graph: Built a local k-NN graph to measure word fluency.\n"
            explanation += f"3. Symbolic Logic (Z3): Enforced strict grammatical Hard Constraints (Template: {template_pos}).\n"
            explanation += "4. Dual-Objective MaxSAT:\n"
            explanation += "   - Maximized Local Fluency (adjacent word similarity).\n"
            explanation += "   - Maximized Global Relevance (similarity to your prompt).\n\n"
            explanation += "Final Grammatically Valid & Topically Relevant Sentence:\n"
            explanation += final_sentence
            
            self.xai_text.setText(explanation)
            self.plot_sequence("Z3 Optimal Path (Relevance + Fluency)", ordered_indices)
            
        else:
            self.log("❌ Z3 Result: UNSATISFIABLE")
            self.xai_text.setText("Error: Cannot fit the retrieved words into the requested template.")

    def plot_sequence(self, title, sequence_order):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.vectors is not None:
            pca = PCA(n_components=2)
            pts_2d = pca.fit_transform(self.vectors)
            
            for i, j in self.knn_edges:
                ax.plot([pts_2d[i, 0], pts_2d[j, 0]], 
                        [pts_2d[i, 1], pts_2d[j, 1]], 'k-', alpha=0.1, zorder=1)
            
            for idx in range(len(sequence_order)-1):
                i = sequence_order[idx]
                j = sequence_order[idx+1]
                ax.annotate('', xy=(pts_2d[j, 0], pts_2d[j, 1]), 
                            xytext=(pts_2d[i, 0], pts_2d[i, 1]),
                            arrowprops=dict(arrowstyle="->", color='red', lw=3))
                
            colors = ['red'] * len(self.vectors)
            for idx in sequence_order:
                colors[idx] = 'green'
                
            ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=colors, s=150, zorder=2)
            
            for i, txt in enumerate(self.metadata):
                pos = self.pos_tags[i] if self.pos_tags else ""
                ax.annotate(f"{txt}\n({pos})", (pts_2d[i, 0], pts_2d[i, 1]), 
                           textcoords="offset points", xytext=(5,10), fontsize=9, weight='bold')
                
        ax.set_title(title)
        ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGNeuroSymbolicApp()
    window.show()
    sys.exit(app.exec())