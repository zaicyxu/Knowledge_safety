# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: main_none_rag.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.3.2
@Description: Verification of RAG system.
"""

import json
import pickle
import re
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
import configuration
from sklearn.decomposition import PCA
from function_call import SystemTools, FactGenerator
from google import genai as gen


# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel(configuration.GENERATION_MODEL)
# client = gen.Client(api_key=configuration.GEMINI_API_KEY)


class Neo4jRAGSystem:
    def __init__(self, uri, user, password, cache_file=configuration.CACHE_FILE):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()
        self.client = gen.Client(api_key=configuration.GEMINI_API_KEY_1)

    def close(self):
        self.driver.close()

    def load_cache(self):
        """Load embedding cache from pickle file."""
        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}

    def save_cache(self):
        """Save embedding cache to pickle file."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        Ensure both vectors are NumPy arrays before computation.
        """
        if isinstance(vec1, str):
            try:
                vec1 = json.loads(vec1)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec1}")
                return 0.0
        if isinstance(vec2, str):
            try:
                vec2 = json.loads(vec2)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec2}")
                return 0.0

        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def update_embeddings(self):
        """Update missing embeddings for nodes in the Neo4j database."""
        with self.driver.session() as session:
            # inside update_embeddings()
            cypher_query = """
                MATCH (n)
                WHERE any(lbl IN labels(n) WHERE lbl IN $target_labels)
                  AND (n.embedding IS NULL OR n.embedding = [])
                RETURN elementId(n) AS id, n.name AS name, elementId(n) AS node_id
            """
            target_labels = configuration.TARGET_LABELS

            records = session.run(cypher_query, target_labels=target_labels)

            for record in records:
                text = record['name']
                embedding = self.get_embedding(text)
                if embedding:
                    update_query = """
                    MATCH (n) WHERE elementId(n) = $node_id
                    SET n.embedding = $embedding
                    """
                    session.run(update_query, node_id=record['node_id'], embedding=embedding)

            print("Embeddings updated successfully.")

    def get_embedding(self, text):
        """Generate embedding using Gemini API or fetch from cache."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            response = genai.embed_content(
                model=configuration.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query"
            )
            embedding = response["embedding"]
            self.embedding_cache[text] = embedding
            self.save_cache()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def extract_keywords(self, query_text, stop_words=configuration.STOP_WORDS, similarity_threshold=configuration.KEYWORD_SIMILARITY_THRESHOLD):
        tokens = re.findall(r'\b\w+\b', query_text)
        tokens = [t for t in tokens if
                  t.lower() not in stop_words]
        tokens = [t for t in tokens if len(t) >= 1]
        # add bigrams
        bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens) - 1)]
        candidates = list(dict.fromkeys(tokens + bigrams))

        q_emb = self.get_embedding(query_text)
        if q_emb is None:
            return [c.lower() for c in candidates[:5]]

        scored = []
        for c in candidates:
            emb = self.get_embedding(c)
            if emb is None:
                continue
            sim = self.cosine_similarity(emb, q_emb)
            scored.append((c, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            return [c.lower() for c in candidates[:5]]
        # return top items with sim >= threshold OR top 5 as fallback
        kws = [w for w, s in scored if s >= similarity_threshold]
        return [w.lower() for w in (kws if kws else [x[0] for x in scored[:5]])]

    def compute_semantic_diversity(self, query_text):
        """
        Compute the semantic diversity of a given query by analyzing the variance in word embeddings.
        A higher variance indicates a more diverse semantic meaning in the query.
        """
        stopwords = configuration.STOP_WORDS
        words = [word for word in re.findall(r'\b\w+\b', query_text.lower())
                 if word not in stopwords]

        if len(words) < configuration.MIN_WORDS_FOR_SEMANTIC_ANALYSIS:
            return 0.4

        # Retrieve embeddings for up to 10 words.
        embeddings = np.array([self.get_embedding(word) for word in words[:configuration.MAX_WORDS_FOR_EMBEDDING]])

        # Apply PCA to reduce dimensionality and analyze variance.
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        variance = np.sum(np.var(reduced, axis=0))

        # Normalize variance using a maximum expected variance value.
        max_variance = configuration.MAX_VARIANCE
        return min(variance / max_variance, 1.0)

    def determine_query_depth(self, query_text):
        """
        Determine the appropriate query depth based on semantic and syntactic complexity.
        """
        semantic_score = self.compute_semantic_diversity(query_text)
        syntax_score = self.compute_syntactic_complexity(query_text)

        # Weighted combination of semantic and syntactic complexity.
        combined_score = (configuration.SEMANTIC_WEIGHT * semantic_score +
                          configuration.SYNTACTIC_WEIGHT * syntax_score)

        if combined_score < 0.2:
            depth = 1
        elif 0.2 <= combined_score < 0.4:
            depth = 2
        elif 0.4 <= combined_score < 0.6:
            depth = 3
        elif 0.6 <= combined_score < 0.8:
            depth = 4
        else:
            depth = 5

        if re.search(r'\b(trace|chain|path|flow|pipeline|scenario|environment|context)\b', query_text, re.I):
            depth = max(depth, 5)

        return depth

    def compute_syntactic_complexity(self, query_text):
        """
        Compute syntactic complexity based on sentence structure.
        """
        words = query_text.split()

        # Normalize sentence length to a max of 15 words.
        length_score = min(len(words) / 15, 1.0)

        wh_words = configuration.WH_WORDS
        wh_score = min(sum(1 for word in words if word.lower() in wh_words) / 2, 1.0)

        # Count clauses
        clause_count = len(re.findall(r',|;| but | however | although ', query_text))
        clause_score = min(clause_count / 2, 1.0)

        # Compute weighted complexity score.
        return (length_score * configuration.LENGTH_WEIGHT +
                wh_score * configuration.WH_WORD_WEIGHT +
                clause_score * configuration.CLAUSE_WEIGHT)

    def local_extract_domain_terms(self, req_text):
            """
            Local replica of extract_svo() from domain.py,
            but defined inside this function so no external import is needed.
            """
            prompt = """
            You are given a requirement sentence related to Autonomous Driving Systems.
            Extract domain-specific technical phrases.

            Rules:
            - Include only meaningful technical or domain-specific phrases.
            - Exclude generic functional words ("shall", "ensure", etc.).
            - Prefer noun phrases or short domain expressions.
            - Each phrase must appear exactly in the text.
            - No duplicates.
            - Ignore content inside brackets.
            - Output ONLY JSON array of strings.
            """

            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt + "\n\n" + req_text,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json"
                    }
                )
                return [t.lower() for t in json.loads(response.text)]
            except Exception as e:
                print(f"[WARN] domain-term extraction failed: {e}")
                return []

    def retrieve_relevant_entities(self, query_text,
                                   top_k=configuration.TOP_K_RESULTS,
                                   similarity_threshold=configuration.SIMILARITY_THRESHOLD):
        """
        RAG disabled for validation experiments.
        Return empty list so generate_answer will proceed without DB facts.
        """
        print("[INFO] RAG retrieval disabled for this run (retrieve_relevant_entities returns []).")
        return []

    def _to_prolog_atom(self, text, fallback=None):
        """
        Convert a given value into a safe Prolog atom.
        """
        # Choose fallback if text is None
        if text is None:
            if fallback is None:
                return "unknown"
            text = str(fallback)

        # Ensure string
        if not isinstance(text, str):
            text = str(text)

        # basic normalization
        raw = text.strip()
        if raw == "":
            raw = fallback if fallback is not None else "unknown"
            raw = str(raw)

        # safe token: lowercase, spaces -> underscores, non-alnum -> underscore
        token = raw.lower().replace(" ", "_")
        token = re.sub(r'[^a-z0-9_]', '_', token)

        # ensure it starts with letter or underscore for bare atom
        if re.match(r'^[a-z_]\w*$', token):
            return token

        # fallback: return quoted atom, escape single quotes inside
        escaped = raw.replace("'", "\\'")
        return f"'{escaped}'"

    def _safe_raw(self, text):
        """
        Keep raw text from DB, but wrap in quotes if it contains spaces or special characters,
        so that Prolog can still parse it correctly.
        """
        if text is None:
            return "unknown"
        text = str(text)
        if re.match(r'^[a-zA-Z0-9_]+$', text):
            return text
        return f"'{text}'"

    def _label_to_prolog_predicate(self, label):
        """
        Convert a graph label or relation name into a safe Prolog predicate name.
        """
        if not label:
            return "unknown"
        s = str(label).strip().lower().replace(" ", "_")
        s = re.sub(r'[^a-z0-9_]', '_', s)
        if not re.match(r'^[a-z_]\w*$', s):
            return f"'{label}'"
        return s

    def call_llm_for_structured_facts(self, user_question, current_prolog_facts):
        """
        Use function-calling style: ask LLM to return structured JSON describing
        new components/requirements/rules to be added.
        """
        schema_instruction = """
        Return JSON with up to three keys: components, requirements, rules.
        components: list of {name, sensors[], algorithms[], actuators[]}
        requirements: list of {req_id, target, sensors[], algorithms[], models[]}
        rules: list of {head, body[]} where head and body contain identifiers, DO NOT emit raw prolog punctuation.
        """

        prompt = f"""
        CURRENT_PROLOG_FACTS:
        {current_prolog_facts}

        USER_QUESTION:
        {user_question}

        INSTRUCTIONS:
        {schema_instruction}

        Output only JSON.
        """

        try:
            response = model.generate_content(prompt)
            raw = response.text
            parsed = json.loads(raw)
        except Exception as e:
            print(f"[WARN] LLM structured call failed: {e}")
            return None

        # Basic validation
        if not isinstance(parsed, dict):
            print("[WARN] LLM did not return JSON object.")
            return None

        # Use SystemTools to convert to facts
        tools = SystemTools(FactGenerator())

        new_facts = []
        for comp in parsed.get("components", []):
            try:
                new_facts += tools.add_new_component(
                    name=comp.get("name", "unknown"),
                    sensors=comp.get("sensors", []),
                    algorithms=comp.get("algorithms", []),
                    actuators=comp.get("actuators", [])
                )
            except Exception as e:
                print(f"[WARN] bad component entry: {comp} -> {e}")

        for req in parsed.get("requirements", []):
            try:
                new_facts += tools.add_new_requirement(
                    req_id=req.get("req_id", "REQ_UNKNOWN"),
                    target=req.get("target", "unknown"),
                    sensors=req.get("sensors", []),
                    algorithms=req.get("algorithms", []),
                    models=req.get("models", [])
                )
            except Exception as e:
                print(f"[WARN] bad requirement entry: {req} -> {e}")

        # rules can be handled by templating too, if present
        for r in parsed.get("rules", []):
            head = r.get("head")
            body = r.get("body", [])
            if head and isinstance(body, list):
                body_str = ", ".join(body)
                new_facts.append(f"{head} :- {body_str}.")

        return new_facts

    def generate_answer(self, user_question: str, facts_str: str = "") -> str:
        """
        Generate a structured answer (Dependency Trace) using the LLM.

        """
        # Build the prompt: keep the same structure as you used before, but make the "always infer" rule explicit.
        prompt = f"""
    ROLE:
    You are an expert in automotive safety engineering and autonomous driving systems.
    Your task is to answer the user's question by producing a short, fully-structured Dependency Trace.

    CRITICAL INSTRUCTION (APPLIES ALWAYS):
    The provided knowledge base may be incomplete or empty. YOU MUST GENERATE any missing single-argument facts
    that are reasonably required to answer the question, using general automotive safety and ADS domain knowledge.
    Do not refuse to infer — produce the most plausible single-argument entities when necessary.

    KNOWLEDGE BASE (REFERENCE ONLY — DO NOT DEPEND ON IT):
    {facts_str}

    USER QUESTION:
    {user_question}

    INSTRUCTIONS (STRICT):
    1) Output ONLY a "Dependency Trace" section listing single-argument Prolog facts relevant to the question.
       Valid facts examples (single-argument only): algorithm(Name). model(Name). sensors(Name).
       component(Name). actuators(Name). system_description(Name). ml_flow(Name). ml_safety_requirement(Name).
    2) Each line must be exactly one fact and must end with a period '.'.
    3) Preserve capitalization, spacing, and exact tokenization in generated names (treat names as identifiers).
    4) Do NOT output requirement_* or reqrelated_* predicates, do NOT output multi-argument relations.
    5) Do NOT provide natural-language explanations.
    6) If you truly cannot infer any plausible entity, output exactly:
       Dependency Trace
       No matching element found.
       Final Answer:

    OUTPUT FORMAT (STRICT):
    Dependency Trace
    <one-fact-per-line, ending with '.'>

    Final Answer:
    """

        raw_response = ""
        # Try preferred call patterns available in your code base
        try:
            # prefer a class wrapper if present
            if hasattr(self, "call_llm") and callable(getattr(self, "call_llm")):
                resp = self.call_llm(prompt)
                # call_llm may return a string or an object with `.text`
                raw_response = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
            # fallback to self.model.generate_content (common pattern in your code)
            elif hasattr(self, "model") and hasattr(self.model, "generate_content"):
                resp = self.model.generate_content(prompt)
                raw_response = resp.text if hasattr(resp, "text") else str(resp)
            # another fallback: self.client.generate
            elif hasattr(self, "client") and hasattr(self.client, "generate"):
                resp = self.client.generate(prompt)
                # adapt for different client return shapes
                if isinstance(resp, str):
                    raw_response = resp
                else:
                    raw_response = getattr(resp, "text", getattr(resp, "content", str(resp)))
            else:
                # last resort: try a global model variable if defined
                try:
                    resp = globals().get("model").generate_content(prompt)
                    raw_response = resp.text if hasattr(resp, "text") else str(resp)
                except Exception:
                    raw_response = ""
        except Exception as e:
            # keep error text minimal but record for debugging
            print(f"[WARN] generate_answer LLM call failed: {e}")
            raw_response = ""

        # If the LLM returned nothing (highly unlikely given "always infer" instruction),
        # provide a conservative fallback inference to avoid empty outputs.
        if not raw_response or not raw_response.strip():
            fallback = [
                "Dependency Trace",
                # conservative minimal inference: common perception pipeline pieces
                "algorithm(ObjectDetection).",
                "model(YOLOv5).",
                "sensors(Mono Camera).",
                "Final Answer:"
            ]
            raw_response = "\n".join(fallback)

        # Return the raw response string for downstream cleaning/formatting.
        return raw_response

    def generate_with_llm(self, input_text):
        """Generate response using Gemini LLM with output cleaning."""
        try:
            response = model.generate_content(input_text)
            return self.clean_response(response.text)
        except Exception as e:
            return "Sorry, I couldn't generate an answer due to an error."

    def normalize_entity(self, raw_entity):
        """
        Normalize a raw retrieval record into a stable schema
        """
        if not raw_entity:
            return {"node1": {"id": None, "name": "", "type": []},
                    "relations": [], "connected_nodes": [], "source": None}

        node1 = raw_entity.get("node1") or {}
        nid = node1.get("id")
        nname = node1.get("name") or ""
        ntype = node1.get("type") or []
        # ensure type is list
        if isinstance(ntype, (str,)):
            ntype = [ntype]
        elif not isinstance(ntype, (list, tuple)):
            ntype = list(ntype) if ntype is not None else []
        else:
            ntype = list(ntype)

        # connected nodes normalization
        raw_conns = raw_entity.get("connected_nodes") or []
        conns = []
        for cn in raw_conns:
            if not cn:
                continue
            cid = cn.get("id")
            cname = cn.get("name") or ""
            ctype = cn.get("type") or []
            if isinstance(ctype, (str,)):
                ctype = [ctype]
            elif not isinstance(ctype, (list, tuple)):
                ctype = list(ctype) if ctype is not None else []
            else:
                ctype = list(ctype)
            conns.append({"id": cid, "name": cname, "type": ctype})

        relations = raw_entity.get("relations") or []
        if isinstance(relations, (str,)):
            relations = [relations]
        elif not isinstance(relations, (list, tuple)):
            try:
                relations = list(relations)
            except Exception:
                relations = []

        return {
            "node1": {"id": nid, "name": nname, "type": ntype},
            "relations": list(relations),
            "connected_nodes": conns,
            "source": raw_entity.get("source")
        }

    def clean_response(self, text: str) -> str:
        if not text:
            return ""

        s = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln.strip() for ln in s.split('\n') if ln.strip()]

        deps, facts, rules = [], [], []

        dep_pred = re.compile(
            r'^(algorithm|model|sensors?|sensor|actuator|actuators?|component|dataset|system_description)\s*\(.*\)\.?$',
            re.IGNORECASE
        )
        fact_pred = re.compile(r'^requirement_[A-Za-z0-9_]*\s*\(.*\)\.?$', re.IGNORECASE)
        rule_pred = re.compile(r'.*:-.*')
        reqrel_pred = re.compile(r'^reqrelated_[A-Za-z0-9_]*', re.IGNORECASE)

        for ln in lines:
            low = ln.lower()
            if low in ("final answer:", "final answer", "prolog-based facts", "prolog-based rules", "dependency trace"):
                continue

            if rule_pred.search(ln) or reqrel_pred.match(ln):
                rules.append(ln if ln.endswith('.') else ln + '.')
                continue
            if fact_pred.match(ln):
                facts.append(ln if ln.endswith('.') else ln + '.')
                continue
            if dep_pred.match(ln):
                deps.append(ln if ln.endswith('.') else ln + '.')
                continue
            if '(' in ln and ')' in ln:
                deps.append(ln if ln.endswith('.') else ln + '.')
                continue

        def quote_after_comma(line):
            # split at first '('
            if '(' not in line or ')' not in line:
                return line
            head, rest = line.split('(', 1)
            args = rest.rstrip(').').split(',')
            new_args = []
            for a in args:
                core = a.strip().strip("'\"")
                new_args.append(f"'{core}'")
            return f"{head}({', '.join(new_args)})."

        facts = [quote_after_comma(f) for f in facts]
        deps = [quote_after_comma(d) for d in deps]

        out = []
        out.append(" ")
        out.append("Dependency Trace")
        out.extend(deps)
        out.append("")

        # if facts:
        #     out.append("Prolog-Based Facts")
        #     out.extend(facts)
        #     out.append("")
        #
        # if rules:
        #     out.append("Prolog-Based Rules")
        #     out.extend(rules)

        return "\n".join(out)

    def rag_pipeline(self, user_question):
        """Execute the RAG pipeline with embedding-based retrieval and in-context learning."""
        relevant_entities = self.retrieve_relevant_entities(user_question)
        return self.generate_answer(user_question, relevant_entities)


def main():
    rag_system = Neo4jRAGSystem(
        uri=configuration.NEO4J_URI,
        user=configuration.NEO4J_USER,
        password=configuration.NEO4J_PASSWORD
    )
    try:
        print("Updating embeddings...")
        rag_system.update_embeddings()
        user_input = input("Enter your question: ")
        answer = rag_system.rag_pipeline(user_input)
        print("\nFinal Answer:", answer)
    finally:
        rag_system.close()


if __name__ == "__main__":
    main()
