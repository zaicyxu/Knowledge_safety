````markdown
# Knowledge_Safety

A RAG-based Large Language Model Framework for Tracing Requirements to Design Information in Safety-Critical Systems

---

## Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** framework that integrates **graph-based knowledge modeling** with **Large Language Models (LLMs)** to support **requirement analysis and safety assurance** in complex engineering systems, with a focus on **Automated Driving Systems (ADS)**.

The framework addresses the problem of **tracing natural-language requirements to heterogeneous design artifacts** (components, sensors, algorithms, DNN models, and ODD elements) in a **precise, interpretable, and scalable** manner by grounding LLM reasoning in a **graph-structured knowledge base** and enabling **multi-hop retrieval**.

---

## Key Idea

1. **Graph-based Knowledge Base**  
   Design information is modeled as a knowledge graph: nodes represent entities (e.g., components, sensors, algorithms, models, ODD elements) and edges represent domain-specific dependencies and hierarchies.

2. **Retrieval-Augmented Generation (RAG)**  
   Relevant subgraphs are retrieved and injected into the LLM prompt, grounding generation in domain facts and reducing hallucination.

3. **Logic Stratification**  
   Natural-language requirements are decomposed into domain-relevant terms, embedded, and matched to graph entities for accurate retrieval.

---

## Framework Architecture

> **[Framework Figure Placeholder]**  
> `![Framework Overview](<img width="3928" height="2152" alt="image" src="https://github.com/user-attachments/assets/3697267a-8f1b-481d-80af-f437861c2a8a" />
)`


**Workflow:**
1. Construct a graph-based knowledge base from domain and system design artifacts.  
2. Index entities with embeddings; extract key terms from requirements (logic stratification).  
3. Retrieve relevant entities and multi-hop dependencies from the graph.  
4. Inject retrieved knowledge into prompts (ICL) and generate structured outputs via the LLM.

---

## Main Contributions

- Graph-based RAG for requirement-to-design traceability in safety-critical systems.  
- Structured, interpretable outputs with explicit dependency chains.  
- Logic stratification to reduce ambiguity in natural-language requirements.  
- Empirical gains over: LLM without RAG, unstructured RAG, prompt-only (CoT), and RAG without logic stratification.

---

## Repository Structure

```text
Knowledge_Safety/
├── data/
├── graph/
├── retrieval/
├── llm/
├── interface/
├── figures/
├── Test_Query_without_requirement.sql
├── configuration.py
├── main_rag_test.py
└── README.md
````

---

## Example of the Input Questions (Requirements)

1. **The object detection module shall reliably detect pedestrians in urban environments.**
2. **The perception system shall ensure robust operation under adverse weather conditions.**
3. **The braking control component shall meet functional safety requirements for emergency scenarios.**

These requirements are used as input to the RAG-based LLM to retrieve relevant graph entities and generate structured, interpretable responses.

---

## How to Run the Test

### Step 1: Build the Knowledge Graph in Neo4j

1. Start Neo4j.
2. Open Neo4j Browser.
3. Execute:

```sql
Test_Query_without_requirement.sql
```

This constructs the nodes, relationships, and schema required by the framework.

---

### Step 2: Configure Database Connection

Edit `configuration.py`:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "your_username"
NEO4J_PASSWORD = "your_password"
```

---

### Step 3: Run the RAG Test

```bash
python main_rag_test.py
```

The script:

1. Loads the graph from Neo4j.
2. Applies logic stratification to extract key terms.
3. Retrieves relevant entities and multi-hop subgraphs.
4. Builds RAG prompts.
5. Generates structured outputs via the LLM.

---

### Step 4: Test with Example Requirements

Use the requirements in **“Example of the Input Questions (Requirements)”** as test inputs. The framework returns structured links to components, algorithms, models, sensors, and ODD elements.

---

## Notes

* Ensure Neo4j is running before executing `main_rag_test.py`.
* The database schema must be created using `Test_Query_without_requirement.sql`.
* Outputs are generated using graph-based retrieval combined with RAG.

---

## Reproducibility

Following the steps above reproduces the requirement-to-design tracing described in the paper.

---

## Citation

```bibtex
@article{su2025ragads,
  title   = {A RAG-based Large Language Model Framework for Tracing Requirements to Design Information of Automated Driving Systems},
  author  = {Su, Peng and Xu, Rui and Huang, Jiacai and Chen, Dejiu},
  journal = {TBD},
  year    = {2025}
}
```

---

## License

MIT License.

---

## Contact

* Rui Xu — [rxu@kth.se](mailto:rxu@kth.se)
* Peng Su — [pengsu@njit.edu.cn](mailto:pengsu@njit.edu.cn)

```
```
