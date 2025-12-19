# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@Project Name: Knowledge_safety

@File Name: configuration.py
@Software: Python
@Time: Sep/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.3.2
@Description: Store sensitive information and hyperparameters
"""

# API Keys
# GEMINI_API_KEY = "AIzaSyBdTPqt4RpQvOc676Z1v_OuEkDsqhrJd9k"
GEMINI_API_KEY_1 = "AIzaSyBdTPqt4RpQvOc676Z1v_OuEkDsqhrJd9k"
GEMINI_API_KEY = "AIzaSyCg8PHoIJhvIukiooFg6b7K2Bi8r-EkhEQ"

# Neo4j Database Credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_BLOOM_URI = "http://localhost:7474/bloom"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# Model Configuration
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash"

# Cache Configuration
CACHE_FILE = "embedding_cache.pkl"

# Retrieval Hyperparameters
# Similarity thresholds
SIMILARITY_THRESHOLD = 0.35
KEYWORD_SIMILARITY_THRESHOLD = 0.35

# Query parameters
DEFAULT_QUERY_DEPTH = 4
MAX_RETRIEVAL_RESULTS = 200
TOP_K_RESULTS = 10

# Keyword extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "and", "or", "of", "to", "in",
    "for", "with", "on", "at", "by", "from", "shall", "be", "all", "any"
}

WH_WORDS = {"detect", "identify", "update", "perform", "integrate",
           "analyze", "maintain", "achieve", "support", "activate"}

# Semantic Analysis Parameters
# Semantic diversity
MAX_VARIANCE = 4.0
MIN_WORDS_FOR_SEMANTIC_ANALYSIS = 5
MAX_WORDS_FOR_EMBEDDING = 10

# Query depth determination weights
SEMANTIC_WEIGHT = 0.6
SYNTACTIC_WEIGHT = 0.4

# Syntactic complexity weights
LENGTH_WEIGHT = 0.5
WH_WORD_WEIGHT = 0.3
CLAUSE_WEIGHT = 0.2

# Neo4j Labels Configuration
TARGET_LABELS = [
    "Object_Detection", "Object_Tracking", "Semantic_Segmentation",
    "Lane_Detection", "Trajectory_Prediction", "Space_Detection",
    "ML_Flow", "ML_Safety_Requirement", "System_Description",
    "System_Safety_Requirement", "Sensors", "algorithms", "actuators",
    "Environment", "Condition", "Road_Type", "Vehicle", "Pedestrian"
]

DESIRED_LABELS = {
    "Object_Detection", "Object_Tracking", "Semantic_Segmentation",
    "Lane_Detection", "Trajectory_Prediction", "Space_Detection",
    "Sensors", "algorithms", "ML_Flow", "Environment", "Condition"
}

# LLM Prompt Configuration
MAX_PROLOG_FACTS_DISPLAY = 50  # Limit facts display in prompt to avoid token limits

# Query Intent Analysis
QUERY_INTENT_ENTITIES = [
    "ML_Flow", "ML_Safety_Requirement", "Sensors", "System_Description",
    "System_Safety_Requirement", "actuators", "algorithms", "functional", "functionalility"
]

QUERY_INTENT_RELATIONSHIPS = [
    "NEXT", "Input", "Output", "Consist", "Include", "Serve", "Collect_Data", "Dataflow"
]