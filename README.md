# Agentic Titanic ML Pipeline
Example of a basic naive implementation of an agentic workflow in the Titanic Dataset EDA Pipeline.

The nature of being a basic pipeline limits how adaptive agentic functionality could be implemented. So maybe this wasn't a great example but it helped ask some important questions and get closer to a concrete vision.

> [!NOTE]
> Due to time constraintes, most tool calls and steps are hardcoded and don't benefit from adaptive agentic capability (e.g. the features `boat` and `body` are hardcoded to be dropped)

## Usage
First, set up your OPENAI api key in the environment where you will run the project. This is used for the LLM agentic tool-calling
```bash
export OPENAI_API_KEY="your-api-key"
```

the run 
```bash
python3 agentic_ml_pipeline.py
```