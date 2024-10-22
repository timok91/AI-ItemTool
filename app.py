import streamlit as st

# Page config
st.set_page_config(
    page_title="AI Item-Entwicklungs-Tool",
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px

@st.cache_resource
def load_models():
    try:
        with st.spinner('Lade Modelle... Dies kann einen Moment dauern.'):
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
            bert_model = BertModel.from_pretrained('bert-base-german-cased')
            sbert_model = SentenceTransformer('deutsche-telekom/gbert-large-paraphrase-cosine')
            return bert_tokenizer, bert_model, sbert_model
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle: {str(e)}")
        return None, None, None

@st.cache_data
def bert_sentence_embedding(sentence, _model, _tokenizer):
    inputs = _tokenizer(
        sentence,
        return_tensors='pt',
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512
    )
    
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = _model(**inputs)
    
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    attention_mask = attention_mask.squeeze(0)
    
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 0)
    sum_mask = torch.clamp(mask_expanded.sum(0), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings.numpy()

# Your default values
DEFAULT_CONSTRUCT = """Emotionsregulation beschreibt den Prozess, durch den Individuen das Erleben, die Intensität, die Dauer, den Zeitpunkt und den Ausdruck von aktivierten Emotionen beeinflussen. Durch Emotionsregulation können pos. und neg. Emotionen verstärkt (Verstärkung), aufrechterhalten oder abgeschwächt werden. Emotionsregulation kann somit als eine Sammlung von kogn. und verhaltensbasierten Strategien zur Beseitigung, Aufrechterhaltung und Veränderung von emot. Erleben und Ausdruck aufgefasst werden. Damit sind generell alle Prozesse gemeint, welche die spontane Entfaltung von Emotionen beeinflussen im Hinblick darauf, welche Emotionen wir haben, wann wir diese haben und wie wir diese erleben und im Verhalten (z. B. Gestik, Mimik) zum Ausdruck bringen. Die Intensität von sowohl pos. als auch neg. Emotionen kann in jede Richtung beeinflusst werden. In der psychol. Emotionsregulation-Forschung interessiert jedoch meist die Verringerung neg. Emotionen: effektive Emotionsregulation besteht demnach darin, pos. Emotionen aufrechtzuerhalten und neg. Emotionen zu verringern."""  

DEFAULT_QUESTIONS = [
    "Im Ärger werde ich manchmal auch lauter.",
    "Wenn es die Situation erfordert, kann ich nach außen hin meine wahren Gefühle verbergen.",
    "Es fällt mir leicht, meine Gefühle bewusst zu verändern.",
    "Wenn ich einmal in schlechter Stimmung bin, kann ich diese immer bewusst verbessern.",
    "Wenn ich will, kann ich mich in eine gute Stimmung bringen.",
    "Selbst starke Erregung und Wut kann ich nach außen besser verbergen als andere.",
    "Wenn ich gereizt und zornig bin, kann ich mich besser beherrschen als andere.",
    "Es fällt mir schwer, meine Gedanken und Emotionen zu kontrollieren, wenn es stressig wird.",
    "Ich habe häufig unkontrollierte Gefühlsausbrüche.", # neues testitem
    "Ich bin ein Biologe." # non-sense testitem
    ]

def main():
    st.title("AI Item-Entwicklungs-Tool")
    
    # Load models
    tokenizer, model, sbert_model = load_models()

    if tokenizer is None or model is None or sbert_model is None:
        st.error("Fehler beim Laden der Modelle. Bitte laden Sie die Seite neu, um es erneut zu versuchen.")
        st.stop()
   
    # Sidebar for instructions
    with st.sidebar:
        st.header("Anleitung")
        st.write("""
        1. Geben Sie die Konstruktdefinition ein
        2. Fügen Sie Items hinzu
        3. Klicken Sie auf 'Analyse starten'
        """)
   
    # Main content area
    col1, col2 = st.columns([2, 1])
   
    with col1:
        st.header("Konstruktdefinition")
        construct = st.text_area(
            "Definition eingeben:",
            value=DEFAULT_CONSTRUCT,
            height=200
        )

    # Questions/Items input
    st.header("Items")
   
    # Initialize session state for questions if not exists
    if 'questions' not in st.session_state:
        st.session_state.questions = DEFAULT_QUESTIONS.copy()
   
    # Add/remove item buttons
    col3, col4 = st.columns([1, 5])
    with col3:
        if st.button("➕ Item hinzufügen"):
            st.session_state.questions.append("")
   
    # Display question input fields
    for i, question in enumerate(st.session_state.questions):
        col_del, col_input = st.columns([1, 7])
        with col_del:
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.questions.pop(i)
                st.rerun()
        with col_input:
            st.session_state.questions[i] = st.text_input(
                f"Item {i+1}",
                value=question,
                key=f"q_{i}"
            )

    # Add analysis button and results display
    if st.button("🔍 Analyse starten", type="primary"):
        if not st.session_state.questions:
            st.error("Bitte fügen Sie mindestens ein Item hinzu.")
        else:
            with st.spinner("Analysiere Items..."):
                # Get current questions from session state
                questions = [q for q in st.session_state.questions if q.strip()]
                
                # Generate embeddings for construct
                construct_embedding_bert = bert_sentence_embedding(construct, model, tokenizer)
                construct_embedding_sbert = sbert_model.encode(construct, normalize_embeddings=True)
                
                # Generate embeddings for questions
                bert_embeddings = [bert_sentence_embedding(q, model, tokenizer) for q in questions]
                sbert_embeddings = sbert_model.encode(questions, normalize_embeddings=True)
                
                # Calculate similarities
                similarities_bert = [abs(cosine_similarity([construct_embedding_bert], [embedding])[0][0]) 
                                   for embedding in bert_embeddings]
                similarities_sbert = [abs(cosine_similarity([construct_embedding_sbert], [embedding])[0][0])
                                    for embedding in sbert_embeddings]
                
                # Create tabs for results
                tab1, tab2, tab3 = st.tabs(["BERT Ergebnisse", "SBERT Ergebnisse", "Item Heatmap"])
                
                with tab1:
                    st.subheader("Top 5 Items (BERT)")
                    results_bert = list(zip(questions, similarities_bert))
                    results_bert.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (question, similarity) in enumerate(results_bert[:5], 1):
                        st.markdown(f"**{i}. Item:** {question}")
                        st.progress(float(similarity))
                        st.markdown(f"Ähnlichkeit: {similarity:.4f}")
                        st.divider()
                
                with tab2:
                    st.subheader("Top 5 Items (SBERT)")
                    results_sbert = list(zip(questions, similarities_sbert))
                    results_sbert.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (question, similarity) in enumerate(results_sbert[:5], 1):
                        st.markdown(f"**{i}. Item:** {question}")
                        st.progress(float(similarity))
                        st.markdown(f"Ähnlichkeit: {similarity:.4f}")
                        st.divider()

                with tab3:
                    st.subheader("Paarweise Ähnlichkeiten zwischen Items")
                    
                    # Calculate pairwise similarities
                    n_items = len(questions)
                    sbert_similarity_matrix = np.zeros((n_items, n_items))
                    
                    # Calculate SBERT similarities
                    for i in range(n_items):
                        for j in range(n_items):
                            similarity = abs(cosine_similarity([sbert_embeddings[i]], [sbert_embeddings[j]])[0][0])
                            sbert_similarity_matrix[i, j] = similarity
                    
                    # Create heatmap using plotly
                    import plotly.express as px
                    
                    fig = px.imshow(
                        sbert_similarity_matrix,
                        labels=dict(x="Items", y="Items", color="Ähnlichkeit"),
                        x=questions,  # Use actual questions instead of numbers
                        y=questions,  # Use actual questions instead of numbers
                        color_continuous_scale=[[0, "white"], [1, "#26358B"]],
                        aspect="auto"
                    )

                    fig.update_layout(
                        width=1000,  # Increased width to accommodate text
                        height=1000,  # Increased height to accommodate text
                        title="Heatmap der Item-Ähnlichkeiten",
                        xaxis_tickangle=-45  # Angle the text for better readability
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    

if __name__ == "__main__":
    main()