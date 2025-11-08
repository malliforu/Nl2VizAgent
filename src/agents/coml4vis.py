# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import warnings

import pandas as pd

# from coml import CoMLAgent, describe_variable

from coml.core import CoMLAgent
from coml.prompt_utils import describe_variable
from langchain.chat_models.base import BaseChatModel
from .agent import Agent, ChartExecutionResult
from .utils import show_svg


# def read_table(name, url, format):
#     code = f"{name}_dataset = pd.read_csv('{url}')"
#     print("CODE:")
#     print(code)
#     variable_description = {}
#     exec(code)
#     exec(
#         f"variable_description['{name}_dataset'] = describe_variable({name}_dataset, dataframe_format='{format}', pandas_description_config=dict(max_rows=10))"
#     )
#     return code, variable_description


def read_table(name, url, format):
    # Load dataset into a local variable
    dataset_var_name = f"{name}_dataset"
    df = pd.read_csv(url)

    # Store the dataset in locals() so you can reference it dynamically later if needed
    locals()[dataset_var_name] = df  

    # Create description
    variable_description = {
        dataset_var_name: describe_variable(
            df,
            dataframe_format=format,
            pandas_description_config=dict(max_rows=10)
        )
    }

    # Generate code string (for logging/debugging only)
    code = f"{dataset_var_name} = pd.read_csv('{url}')"
    return code, variable_description

class CoML4VIS(Agent):
    def __init__(self, llm: BaseChatModel, config: dict = None):
        num_examples = 1
        prompt_version = "matplotlib"
        if config:
            if "num_examples" in config:
                num_examples = min(config["num_examples"], 4)
            if "library" in config and config["library"] in [
                "matplotlib",
                "seaborn",
            ]:
                prompt_version = config["library"]

        self.coml = CoMLAgent(
            llm, num_examples=num_examples, prompt_version=prompt_version
        )

    def pre_code(self, tables: list[dict], chart_lib: str, table_format: str = "coml"):
        codes = ["import pandas as pd\nimport matplotlib.pyplot as plt\n"]
        variable_descriptions = {}
        if chart_lib == "seaborn":
            codes[-1] += "import seaborn as sns\n"

        for url in tables:
            name = url.split("/")[-1].split(".")[0]
            code, variable_description = read_table(name, url, table_format)
            codes.append(code)
            variable_descriptions.update(variable_description)
        return codes, variable_descriptions

    def generate(self, nl_query: str, tables: list[str], config: dict):
        print("[CoML4VIS] generate started =====")
        library = config["library"]
        table_format = config["table_format"] if "table_format" in config else "coml"

        pre_codes, variable_descriptions = self.pre_code(tables, library, table_format)
        generating_context = self.coml.generate_code(
            nl_query, variable_descriptions, pre_codes
        )
        try:
            generating_context = self.coml.generate_code(
                nl_query, variable_descriptions, pre_codes
            )
            generate_code = generating_context["answer"]

            context = {"tables": tables, "library": library}
            print("[CoML4VIS] generate completed =====")
            return "\n".join(pre_codes) + "\n" + generate_code, context
        except Exception:
            warnings.warn(str(sys.exc_info()))
        print("[CoML4VIS] generate failed =====")
        return None, None

    def execute(self, code: str, context: dict, log_name: str = None):
        print("[CoML4VIS] execute started =====")
        tables = context["tables"]
        library = context["library"]

        global_env = {"svg_string": None, "show_svg": show_svg, "svg_name": log_name}
        code += "\nsvg_string = show_svg(plt, svg_name)"
        try:
            # print("CoML4VIS code:\n", code)
            code = code.replace("plt.show()", '')
            exec(code, global_env)
            svg_string = global_env["svg_string"]
            print("[CoML4VIS] Executed successfully")
            return ChartExecutionResult(status=True, svg_string=svg_string)
        except Exception as exception_error:
            try:
                # handle old version
                codes, variable_descriptions = self.pre_code(tables, library)
                exec("\n".join(codes) + "\n" + code, global_env)
                svg_string = global_env["svg_string"]
                print("[CoML4VIS] Executed successfully")
                return ChartExecutionResult(status=True, svg_string=svg_string)
            except Exception as exception_error:
                error_msg = str(exception_error)
                print(f"[CoML4VIS] Execution failed: {error_msg}")
                return ChartExecutionResult(status=False, error_msg=error_msg)
