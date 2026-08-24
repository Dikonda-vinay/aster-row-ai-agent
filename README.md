\# Aster \& Row Customer Support AI Agent



An AI-powered customer support agent built using Retrieval-Augmented Generation (RAG), local semantic embeddings, and the Groq LLM API.



The agent answers customer support questions using a controlled company knowledge base and can safely perform customer-safe order lookups.



\## Features



\- Answers customer questions using the company knowledge base

\- Semantic retrieval using Sentence Transformers

\- Uses only active and official policies for authoritative answers

\- Handles conflicting policy documents safely

\- Protects against prompt injection

\- Provides safe order lookups

\- Protects private customer information

\- Supports multi-turn conversations

\- Provides source citations

\- Recommends human support when required

\- Handles missing or insufficient information without guessing



\## How It Works



The agent follows this pipeline:



User Question

|

v

Session / Conversation Context

|

v

Semantic Knowledge Retrieval

|

v

Relevant Policy Passages

|

v

Grounded LLM Response

|

+---- Order Lookup Tool

|

v

Customer-Safe Response



\## Technologies



\- Python

\- Groq API

\- Sentence Transformers

\- NumPy

\- PyYAML

\- pytest

\- Retrieval-Augmented Generation (RAG)



\## Knowledge Base



The knowledge base contains company policies stored as Markdown files.



It covers:



\- Returns

\- Shipping

\- Warranty

\- Damaged or wrong items

\- TrailPlus membership

\- Product care

\- International shipping

\- Order changes and cancellations

\- Gift cards and price adjustments



Each policy document contains metadata such as:



\- Status

\- Effective date

\- Last reviewed date

\- Policy authority



Only documents marked as active and official are treated as authoritative.



\## Order Lookup



The agent includes an order lookup tool for customer-safe order information.



It can provide:



\- Order status

\- Carrier

\- Tracking information

\- Estimated delivery date



Sensitive internal information is not disclosed, including:



\- Customer email

\- Shipping address

\- Internal notes

\- Risk scores

\- Fraud-related information



\## Evaluation



The project includes automated tests and behavior-level evaluation cases.



Run the tests with:



python -m pytest -q



Current test result:



17 passed



The evaluation covers:



\- Standard return policy

\- TrailPlus return policy

\- Damaged final-sale items

\- International shipping

\- Unsupported countries

\- Order lookup

\- Missing order IDs

\- Cancelled orders

\- Unknown orders

\- Privacy protection

\- Warranty questions

\- Prompt injection

\- Insufficient information

\- Conflicting official sources

\- Human handoff behavior



\## Known Limitations



\- The agent depends on the supplied knowledge base for company-specific information.

\- It cannot directly perform refunds, cancellations, replacements, or other customer-service actions.

\- Some conflicting or incomplete policy situations require human support.

\- Order information comes from the supplied test dataset.

\- The application currently uses a terminal-based CLI rather than a web interface.

\- The quality of responses depends partly on the quality of retrieved knowledge-base passages.



\## Setup



Create a virtual environment:



python -m venv .venv



Activate it on Windows:



.venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt



Create a .env file using .env.example and add the required API configuration.



Do not commit the .env file or API keys.



\## Run



Start the agent:



python cli.py



For debug mode:



python cli.py --debug



\## Testing



Run:



python -m pytest -q



Expected result:



17 passed



\## Project Structure



aster-row-agent/

|

+-- app/

+-- data/

+-- evaluation/

+-- knowledge-base/

+-- tests/

+-- cli.py

+-- requirements.txt

+-- README.md

+-- .env.example

+-- .gitignore



\## Demo

A short demonstration of the Aster & Row customer support agent:

[▶️ Watch the demo video](./demo.mp4)



\## Security



The repository must not contain:



\- API keys

\- Passwords

\- Credentials

\- Private customer information

\- Production customer datasets



The .gitignore file excludes local environment files and generated runtime files.

