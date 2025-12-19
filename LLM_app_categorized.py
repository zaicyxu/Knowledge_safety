# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: LLM_app_categorized.py
@Software: Python
@Time: Nov/2025
@Description: Streamlit GUI for Neo4j RAG Question Answering System.
              Focuses exclusively on displaying Categorized Elements from Neo4j
              with a beautified UI.
"""

import re
import hashlib
import streamlit as st
from main_rag_test import Neo4jRAGSystem
import configuration
from collections import defaultdict

# --- Custom CSS for Academic Visualization ---
CUSTOM_CSS = """
<style>
    /* Global Styles - Academic Paper Theme */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Lato:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
        color: #333;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', serif;
        color: #2c3e50;
    }

    /* Main Container */
    .main {
        background-color: #ffffff;
    }

    /* Title */
    h1 {
        font-weight: 700;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Cards/Sections */
    .academic-card {
        background-color: white;
        border: 1px solid #ddd;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: none; /* Flat look for academic style */
    }

    /* Category Header */
    .category-header {
        font-family: 'Merriweather', serif;
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }

    /* Items List */
    .item-list {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }
    .item-entry {
        padding: 4px 0;
        font-size: 0.95rem;
        border-bottom: 1px dotted #eee;
    }
    .item-entry:last-child {
        border-bottom: none;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        color: #495057;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        border-top: 2px solid #2c3e50;
        color: #2c3e50;
    }

    /* Form Styling */
    .stTextArea textarea {
        border-radius: 4px;
        border: 1px solid #ccc;
        font-family: 'Lato', sans-serif;
    }
    .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        border: none;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    .stButton button:hover {
        background-color: #1a252f;
        color: white;
    }
</style>
"""

def init_rag_system() -> Neo4jRAGSystem:
    """Initialize and cache RAG system in session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = Neo4jRAGSystem(
            uri=configuration.NEO4J_URI,
            user=configuration.NEO4J_USER,
            password=configuration.NEO4J_PASSWORD,
        )
    return st.session_state.rag_system

def clear_chat_history() -> None:
    """Clear conversation history from session_state."""
    st.session_state.history = []

def parse_response(text: str) -> dict:
    """
    Parse backend text to extract the Dependency Trace Elements.
    """
    out = {"elements": "", "final_facts": [], "prolog_rules": ""}
    if not text:
        return out
    
    pattern = re.compile(
        r"Dependency Trace\s*(.*?)\s*Prolog-Based Facts\s*(.*?)\s*Prolog-Based Rules\s*(.*)",
        re.DOTALL
    )
    
    match = pattern.search(text)
    if match:
        out["elements"] = match.group(1).strip()
        out["final_facts"] = match.group(2).strip()
        out["prolog_rules"] = match.group(3).strip()
    
    return out

import graphviz

def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Requirement Analysis - Academic View", layout="wide", page_icon="🎓")
    
    # Inject Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Tracing Requirements to Design Information in Automated Driving Systems")
    st.markdown(
        """
        <div style='margin-bottom: 2rem; color: #555; font-style: italic;'>
        This tool utilizes a Retrieval-Augmented Generation (RAG) framework to analyze system requirements. 
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Initialize
    rag_system = init_rag_system()
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("qa_form"):
        user_input = st.text_area(
            "Input Requirements Specification",
            value="",
            height=120,
            placeholder="e.g. The total latency from sensor data capture to actuator command shall not exceed 200 milliseconds for any safety-critical function."
        )
        submitted = st.form_submit_button("Analyze")

    if submitted:
        clear_chat_history()
        question = user_input.strip()
        if not question:
            st.warning("Please enter a non-empty requirement.")
        else:
            with st.spinner("Processing requirements..."):
                try:
                    raw_answer = rag_system.rag_pipeline(question)
                    parsed = parse_response(raw_answer)

                    st.session_state.history.append({
                        "question": question,
                        "Categoried Elements": parsed["elements"],
                    })
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

    if st.session_state.history:
        st.markdown("---")
        # Render newest first
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            
            # Parse Categories
            if entry["Categoried Elements"]:
                categories = defaultdict(list)
                pattern = re.compile(r"(\w+)\(([^)]+)\)")
                items = entry["Categoried Elements"].split('\n')
                
                for item in items:
                    match = pattern.search(item)
                    if match:
                        category, name = match.groups()
                        categories[category].append(name.strip())
                
                if categories:
                    # Create Tabs for different views
                    tab1, tab2, tab3 = st.tabs(["� Detailed List", "📊 Visual Analysis", "🕸️ Network Graph"])
                    
                    with tab1:
                        st.markdown("### Categorized Elements")
                        # Display categories in a responsive grid using columns
                        cols = st.columns(3) # Fixed 3 columns for academic layout
                        
                        cat_items = list(categories.items())
                        for idx, (category, items) in enumerate(cat_items):
                            with cols[idx % 3]:
                                st.markdown(
                                    f"""
                                    <div class="academic-card">
                                        <div class="category-header">{category}</div>
                                        <ul class="item-list">
                                            {''.join([f'<li class="item-entry">{item}</li>' for item in items])}
                                        </ul>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    with tab2:
                        st.markdown("### Distribution of Elements")
                        # Prepare data for chart
                        data = {"Category": list(categories.keys()), "Count": [len(v) for v in categories.values()]}
                        st.bar_chart(data, x="Category", y="Count", color="#2c3e50")
                        
                        st.markdown("#### Summary Statistics")
                        total_elements = sum(len(v) for v in categories.values())
                        st.write(f"**Total Extracted Elements:** {total_elements}")
                        st.write(f"**Unique Categories:** {len(categories)}")

                    with tab3:
                        st.markdown("### Knowledge Graph Visualization")
                        # Create Graphviz Digraph
                        graph = graphviz.Digraph()
                        graph.attr(rankdir='LR', size='10')
                        graph.attr('node', shape='box', style='filled', fillcolor='#ecf0f1', fontname='Lato')
                        graph.attr('edge', color='#bdc3c7')
                        
                        # Root node
                        graph.node('Requirement', 'Requirement Analysis', shape='ellipse', fillcolor='#2c3e50', fontcolor='white')
                        
                        for category, items in categories.items():
                            # Category node
                            cat_id = f"cat_{category}"
                            graph.node(cat_id, category, shape='folder', fillcolor='#3498db', fontcolor='white')
                            graph.edge('Requirement', cat_id)
                            
                            for item in items:
                                # Item node
                                item_id = f"item_{hash(item)}"
                                graph.node(item_id, item)
                                graph.edge(cat_id, item_id)
                        
                        st.graphviz_chart(graph)

                else:
                    st.info("No categorized elements found in the analysis.")
            else:
                st.info("No elements returned from the analysis.")
if __name__ == "__main__":
    main()
