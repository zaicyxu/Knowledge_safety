
# Knowledge_Safety

A RAG-based Large Language Model Framework for Tracing Requirements to Design Information in Safety-Critical Systems

---

## Overview

This repository provides an implementation of a **Retrieval-Augmented Generation (RAG)** framework that integrates **graph-based knowledge modeling** with **Large Language Models (LLMs)** to support **requirement analysis and safety assurance** in complex engineering systems, particularly **Automated Driving Systems (ADS)**.

In safety-critical system development, a central challenge is how to **systematically trace natural-language requirements to heterogeneous design artifacts** (e.g., components, algorithms, models, sensors, and Operational Design Domain (ODD) elements) in a precise, interpretable, and scalable manner.  
Conventional LLM-based approaches often suffer from hallucination, lack of domain grounding, and limited capability in multi-level reasoning.

To address these issues, this project proposes a framework that:
- Models design information as a **graph-based knowledge base**;
- Retrieves relevant subgraphs via **multi-hop reasoning**;
- Integrates retrieved domain knowledge into LLM prompts using **RAG**;
- Generates **structured and explainable outputs** that explicitly link requirements to design information.

---

## Key Idea

The framework is built on three core concepts:

### 1. Graph-based Knowledge Base
Design information is explicitly modeled as a knowledge graph:
- **Nodes** represent entities such as system components, sensors, algorithms, DNN models, and ODD elements.
- **Edges** represent domain-specific dependencies and hierarchical relations.

This representation enables efficient retrieval and interpretable multi-hop reasoning.

### 2. Retrieval-Augmented Generation (RAG)
Instead of relying solely on the LLM’s internal knowledge, the framework:
- Retrieves the most relevant entities and their dependencies from the knowledge graph;
- Injects this information into the prompt;
- Grounds the LLM’s responses in domain-specific facts, reducing hallucination.

### 3. Logic Stratification for Requirement Analysis
Natural-language requirements are decomposed into domain-relevant terminologies.  
These keywords are embedded and matched against graph entities to locate the most relevant subgraphs for reasoning.

---

## Framework Architecture

The overall workflow of the proposed framework is illustrated below.

> **<img width="3928" height="2152" alt="image" src="https://github.com/user-attachments/assets/32c36156-fabc-44d1-bdea-dd3609d04397" />
**  
```

**Workflow Summary:**

1. **Knowledge Base Construction**  
   Domain and system design artifacts are modeled using a graph-based meta-model and instantiated into a knowledge graph (e.g., in Neo4j).

2. **Indexing and Retrieval**  
   Entities in the graph are embedded for semantic indexing.  
   Input requirements are analyzed via logic stratification to extract key terminologies, which are then matched to graph entities to retrieve relevant subgraphs (including multi-hop dependencies).

3. **LLM-based Reasoning**  
   Retrieved knowledge is injected into structured prompts using In-Context Learning (ICL).  
   The LLM generates structured responses that explicitly trace requirements to design information.

---

## Main Contributions

- **Graph-based RAG for Requirement Traceability**  
  A functional framework that combines knowledge graphs and LLMs to trace requirements to design artifacts in safety-critical systems.

- **Structured and Interpretable Outputs**  
  Generates structured responses that make dependencies between requirements and design information explicit and explainable.

- **Logic Stratification for Domain Queries**  
  Reduces ambiguity in natural-language requirements by extracting and matching domain-specific terminologies.

- **Empirical Validation**  
  Demonstrates improved precision, recall, and F1-score over:
  - LLM without RAG;
  - RAG with unstructured (text-based) knowledge;
  - Prompt engineering (e.g., Chain-of-Thought);
  - RAG without logic stratification.

---

## Repository Structure

```text
Knowledge_Safety/
├── data/                 # Datasets and example requirements
├── graph/                # Knowledge graph schema and construction scripts
├── retrieval/            # Indexing, embedding, and multi-hop retrieval logic
├── llm/                  # Prompt templates and LLM interaction
├── interface/            # Optional UI / visualization components
├── figures/              # Framework and example knowledge graphs
├── Test_Query_without_requirement.sql
├── configuration.py
├── main_rag_test.py
└── README.md
````

---

## Example of the Input Questions (Requirements)

The following examples illustrate typical requirement queries used to evaluate the framework.
Each requirement is written in natural language and aims to trace high-level safety or functional constraints to concrete design information stored in the knowledge graph.

**Examples:**

1. **“The object detection module shall reliably detect pedestrians in urban environments.”**
2. **“The perception system shall ensure robust operation under adverse weather conditions.”**
3. **“The braking control component shall meet functional safety requirements for emergency scenarios.”**

These requirements are used as input to the RAG-based LLM, which retrieves relevant entities and relationships from the graph-based knowledge base and generates structured, interpretable responses.

---

## How to Run the Test

The framework evaluates requirement tracing by querying a **Neo4j-based knowledge graph** and performing RAG-based reasoning with an LLM.

Follow the steps below to reproduce the experiments and test the system with the example requirements.

---

### Step 1: Build the Knowledge Graph in Neo4j

1. Start your Neo4j database service.
2. Open the **Neo4j Browser**.
3. Execute the following Cypher script to construct the database:

```sql
Test_Query_without_requirement.sql
```

This script creates the nodes, relationships, and schema required by the framework, including:

* System components,
* Sensors,
* Algorithms,
* DNN models,
* ODD elements and their dependencies.

---

### Step 2: Configure Database Connection

Edit the Neo4j authentication settings in `configuration.py`:

```python
# configuration.py

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "your_username"
NEO4J_PASSWORD = "your_password"
```

Ensure that the URI, username, and password match your local Neo4j instance.

---

### Step 3: Run the RAG Test

Execute the main test script:

```bash
python main_rag_test.py
```

This script will:

1. Load the knowledge graph from Neo4j.
2. Apply logic stratification to extract key terms from the input requirement.
3. Retrieve the most relevant entities and their multi-hop dependencies.
4. Construct RAG-based prompts with retrieved knowledge.
5. Invoke the LLM to generate structured, explainable outputs.

---

### Step 4: Test with Example Requirements

Use the requirements listed in **“Example of the Input Questions (Requirements)”** as input for testing.
Each requirement will be processed by the framework, and the corresponding design elements (e.g., components, algorithms, models, sensors, ODD elements) will be returned in a structured and interpretable format.

---

## Notes

* Ensure that **Neo4j is running** before executing `main_rag_test.py`.
* The framework assumes that the database schema is constructed exclusively using `Test_Query_without_requirement.sql`.
* All outputs are generated using **graph-based retrieval combined with RAG**, rather than relying solely on the LLM’s internal knowledge.

---

## Reproducibility

By following the steps above, users can reproduce the requirement tracing process described in the paper and verify how the framework systematically links natural-language requirements to system design information in a safety-critical context.

---

## Citation

If you use this work in your research, please cite the corresponding paper:

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

This project is released under the MIT License (or replace with your chosen license).

---

## Contact

For questions, collaborations, or issues, please open an issue on GitHub or contact:

* Rui Xu — [rxu@kth.se](mailto:rxu@kth.se)


```


