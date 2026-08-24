\# Aster \& Row Customer Support AI Agent



An AI-powered customer support agent built using RAG (Retrieval-Augmented Generation), local embeddings, and the Groq LLM API.



\## Features



\- Answers customer questions using the company knowledge base

\- Semantic search using sentence-transformers

\- Uses only active and official policies for authoritative answers

\- Protects against prompt injection

\- Provides safe order lookups

\- Protects private customer information

\- Supports multi-turn conversations

\- Provides source citations

\- Recommends human support when necessary



\## How It Works



The agent follows this pipeline:



User Question

&#x20;    |

&#x20;    v

Knowledge Base Retrieval

&#x20;    |

&#x20;    v

Relevant Policy Documents

&#x20;    |

&#x20;    v

Groq LLM

&#x20;    |

&#x20;    +---- Order Lookup Tool

&#x20;    |

&#x20;    v

Grounded Customer Response



\## Technologies



\- Python

\- Groq API

\- Sentence Transformers

\- NumPy

\- YAML

\- pytest

\- RAG



\## Knowledge Base



The knowledge base contains company policies in Markdown format.



Documents include information about:



\- Returns

\- Shipping

\- Warranty

\- Damaged or wrong items

\- TrailPlus membership

\- Product care

\- International shipping



Documents contain metadata such as:



\- status

\- effective date

\- policy authority



Only documents marked as active and official are treated as authoritative.



\## Order Lookup



The order lookup tool prevents sensitive information from reaching the AI model.



The following internal information is

