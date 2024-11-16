# Startup Idea Generator

## Overview

This project is a simple application that generates startup ideas based on a given topic. It utilizes descriptions of Y Combinator companies as a knowledge base to inspire and suggest new business concepts.

## Dataset

The dataset used is sourced from [Kaggle: Y Combinator Directory](https://www.kaggle.com/datasets/miguelcorraljr/y-combinator-directory/data). It contains descriptions of companies from Y Combinator, which serve as the foundation for generating new ideas.

## Features

- Generate business ideas on specific topics.
- Leverage a knowledge base of successful startups for inspiration.
- Simple API endpoint to ask questions and receive concise answers.

## Example Usage

You can ask the application to generate a startup idea in a particular field:

- **Question**: "Generate me a startup idea in the field of biotech."

## Installation

### Prerequisites

- Python 3.12
- pip package manager
- OpenAI API Key
- LangChain API Key

### Clone the Repository

```bash
git clone https://github.com/miakovlev/AI_sandbox.git
cd AI_sandbox/rag_ycombinator_generator
```

### Setup Environment Variables
Create a `.env` file in the `rag_ycombinator_generator` directory and add your API keys:

```dotenv
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

### Install Dependencies
Install the required Python packages:

```
pip install -r requirements.txt
```

### Prepare the Dataset
1. Download the dataset from Kaggle.
2. Place the CSV file in the database directory and name it 2023-07-13-yc-companies.csv.

### Run the Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage
### Endpoint: /ask
**Method:** POST

Request Body:
```json
{
  "question": "Generate me a startup idea in the field of biotech."
}
```
Response:
```json
{
  "answer": "A startup idea in the biotech field could focus on developing an AI-driven platform that integrates DNA sequencing and synthesis with machine learning to optimize the design of genetically modified organisms for sustainable manufacturing. This platform could help biotech companies rapidly prototype and test new organisms for producing advanced materials, fuels, and therapeutics, significantly reducing time to market. Additionally, it could provide real-time data analysis tools to enhance decision-making in R&D processes."
}
```


**Disclaimer:** This project is an AI sandbox and is not intended for production use. The generated ideas are for inspiration purposes only.