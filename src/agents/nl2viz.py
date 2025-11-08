from typing_extensions import Union, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import pandas as pd
import re
import os
from attr import dataclass
import uuid
import json
from agents import Agent

@dataclass
class ChartExecutionResult:
    status: bool
    svg_string: str
    error_msg: str

class Nl2Viz(Agent):
    def __init__(self, llm, config):
        print("[Nl2VizAgent] Initialized ====")
        self.llm = llm
        self.config = config
        # os.makedirs(self.config.get("output_dir", 'output'), exist_ok=True)
    
    def _get_messages(self, input_prompt: Union[ChatPromptTemplate, List, str]) -> List:
        """Helper method to extract messages from different input types"""
        if isinstance(input_prompt, ChatPromptTemplate):
            return input_prompt.messages
        elif isinstance(input_prompt, list):
            return input_prompt
        elif isinstance(input_prompt, str):
            return [HumanMessage(content=input_prompt)]
        else:
            raise ValueError(f"Unsupported input type: {type(input_prompt)}")
    
    def _load_table_contexts(self, tables):
        print("[Nl2VizAgent] _load_table_contexts started =====")
        table_contexts = {}
        for table_path in tables:
            if table_path.endswith(".csv"):
                df = pd.read_csv(table_path)
                table_contexts[table_path] = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col,dtype in df.dtypes.items()},
                    "sample": df.head(5).to_dict('records'),
                    "shape": df.shape
                }
            elif table_path.endswith('.json'):
                with open(table_path,'r') as f:
                    data = json.load(f)
                table_contexts[table_path] = {
                    "sample": data[:5] if isinstance(data, list) else data,
                    "type": "list" if isinstance(data, list) else "dict"
                }
            else:
                print("[Nl2VizAgent][_load_table_contexts]: Unsupported file ====")
                table_contexts[table_path] = {'error':"Unsupported file format"}
        print("[Nl2VizAgent] _load_table_contexts completed =====")
        return table_contexts

    def _construct_code_generation_prompt(self, nl_query, table_contexts):
        print("[Nl2VizAgent] _construct_code_generation_prompt started =====")
        prompt = f"""
    You are an expert data visualization assistant. Generate Python code to create the visualization described below.

    QUERY: {nl_query}

    AVAILABLE DATA:
    """
        for table_path, context in table_contexts.items():
            prompt += f"\n- Table: {table_path}\n"

            if "error" in context:
                prompt += f" Error Loading table: {context['error']}\n"
                continue

            if 'columns' in context:
                prompt += f" Columns: {', '.join(context['columns'])}\n"
                prompt += f" Data types: {context['dtypes']}\n"
                prompt += f" Shape {context['shape'][0]} rows, {context['shape'][1]} columns\n"
                prompt += f" Sample data:\n"
                for row in context['sample'][:3]:
                    prompt += f" {row}\n"
            elif "sample" in context:
                prompt += f" Sample data: {context['sample']}\n"
        
        prompt += f"""
    REQUIREMENTS:
    1. Use pandas for data loading and manipulation.
    2. Use matplotlib and/or seaborn for visualization.
    3. Create a clear and informative visualization that answers the query.
    4. Include appropriate labels, titles, and legends.
    5. Use theme: "ggplot"
    6. Return SVG format visualization.
    7. IMPORTANT: When loading files, use their FULL PATH as specified above (e.g., "{list(table_contexts.keys())[0]}" not just the filename).

    Please provide only the Python code without explanations. The code should be complete and ready to execute.
    """
        print("[Nl2VizAgent] _construct_code_generation_prompt completed ====")
        return prompt
    
    def generate(self, nl_query, tables=None, config=None):
        print("[Nl2VizAgent] generate started =====")
        if tables:
            table_contexts = self._load_table_contexts(tables)
            prompt = self._construct_code_generation_prompt(nl_query, table_contexts)
            messages = self._get_messages(prompt)
        else:
            messages = self._get_messages(nl_query)
        response = self.llm.invoke(messages)

        # Extract content from AIMessage
        if hasattr(response, "content"):
            # Direct attribute access for AIMessage objects
            generated_code = response.content
        elif isinstance(response, dict) and "content" in response:
            # Dictionary access for dict-like responses
            generated_code = response["content"]
        else:
            # Handle case where response is a string or other format
            generated_code = str(response)

        if "```python" in generated_code:
            code_start = generated_code.find("```python") + 9
            code_end = generated_code.find("```", code_start)
            generated_code = generated_code[code_start:code_end].strip()
            
        # Create context for execution
        context = {
            "tables": tables,
            "table_contexts": table_contexts,
        }
        
        print("[Nl2VizAgent] generate completed =====")
        return generated_code, context

    def execute(self, code, context, log_name=None):
        print("[Nl2VizAgent] execute started =====")

        if log_name is None:
            log_name = f"viz_{uuid.uuid4().hex[:8]}"

        execution_env = {
            # Provide access to common visualization libraries
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
            # Add table data from context
            "tables": context.get("tables", []),
            "table_contexts": context.get("table_contexts", {}),
            # Add utility functions
            "os": os,
            # Add a placeholder for the SVG output
            "svg_output": None,
            "svg_name": log_name
        }

        code = code.replace("plt.show()", "")

        # Add code to capture SVG output
        code_with_svg_capture = f"""
import io
from matplotlib.backends.backend_svg import FigureCanvasSVG

{code}

# Capture SVG output
buf = io.BytesIO()
plt.savefig(buf, format='svg')
svg_output = buf.getvalue().decode('utf-8')
plt.close()
"""

        code_with_svg_capture = code_with_svg_capture.replace("plt.show()", '')
        #removing the piece of code which causing the svg capture to fail
        remove_texts = re.findall('.*plt.close\(.*\).*',code_with_svg_capture)
        for text in remove_texts:
            code_with_svg_capture = code_with_svg_capture.replace(text, '')

        try:
            exec(code_with_svg_capture, execution_env)
            print("[Nl2VizAgent] execute completed =====")
            return ChartExecutionResult(status=True, svg_string=execution_env.get('svg_output', ''), error_msg='No Error')
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[Nl2VizAgent] Execution error: {str(e)}")
            return ChartExecutionResult(status=False, svg_string='', error_msg=f"{str(e)}\n\n{error_details}")

